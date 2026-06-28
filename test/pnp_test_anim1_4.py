#!/usr/bin/env python3
"""
Vision-based gate pose estimation and EKF fusion for drone racing.
Processes a pre-recorded dataset (images + IMU + ground truth) and saves results.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Import the needed modules from the original code
# (Assume they are in the same directory or adjust sys.path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.vision.quad_gate import QuAdGate, GateTracker, GateDetection
from src.vision.pose_estimator import PoseEstimator, GatePose
from src.state.ekf import ExtendedKalmanFilter, EKFState


# ----------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------
class RacingDataset:
    """Loads images, IMU, and ground truth from a dataset folder."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "imgs"
        self.metadata_path = self.data_dir / "metadata.json"
        self.imu_path = self.data_dir / "imu_data.csv"

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load IMU/ground truth CSV
        self.df = pd.read_csv(self.imu_path)
        self.num_frames = len(self.df)

        # Gate information (world frame)
        gate = self.metadata["gate"]
        self.gate_pos_world = np.array(gate["position"], dtype=np.float64)  # [x,y,z]
        self.gate_euler = np.array(gate["euler_xyz_rad"], dtype=np.float64)  # [roll,pitch,yaw]
        self.gate_width = gate["dimensions"][1]
        self.gate_height = gate["dimensions"][2]

        # Camera intrinsics from metadata
        cam = self.metadata["camera"]
        fx = cam["focal_length_mm"] / cam["sensor_width_mm"] * cam["resolution_x"]
        fy = cam["focal_length_mm"] / cam["sensor_height_mm"] * cam["resolution_y"]
        cx = cam["resolution_x"] / 2.0
        cy = cam["resolution_y"] / 2.0
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.image_size = (cam["resolution_x"], cam["resolution_y"])  # (width, height)

        # Gate rotation matrix (world frame)
        self.R_gate_world = R.from_euler("xyz", self.gate_euler, degrees=False).as_matrix()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        img_path = self.image_dir / f"{idx:04d}.jpg"
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Time
        time = row["time"]

        # Ground truth drone pose (world frame)
        pos_world = np.array([row["pos_x"], row["pos_y"], row["pos_z"]], dtype=np.float64)
        # Quaternion in CSV: (w,x,y,z) -> convert to (x,y,z,w)
        q_wxyz = np.array([row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]], dtype=np.float64)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # (x,y,z,w)

        # IMU data (body frame)
        angvel = np.array([row["angvel_x"], row["angvel_y"], row["angvel_z"]], dtype=np.float64)
        accel = np.array([row["accel_x"], row["accel_y"], row["accel_z"]], dtype=np.float64)

        return {
            "image_rgb": image_rgb,
            "time": time,
            "pos_world": pos_world,
            "q_xyzw": q_xyzw,
            "angvel": angvel,
            "accel": accel,
            "image_path": img_path,
        }


# ----------------------------------------------------------------------
# Helper: coordinate transformations
# ----------------------------------------------------------------------
def quat_xyzw_to_scipy(q_xyzw):
    """Convert (x,y,z,w) to scipy Rotation object."""
    return R.from_quat(q_xyzw)


def scipy_to_quat_xyzw(rot):
    """Convert scipy Rotation to (x,y,z,w) quaternion."""
    return rot.as_quat()  # already (x,y,z,w)


def compute_gate_pose_in_camera(dataset, drone_pos, drone_quat_xyzw, R_cam_to_body):
    """
    Compute true gate pose in camera frame from ground truth.
    Returns: (t_cam_gate, R_cam_gate) where R_cam_gate is 3x3 rotation from gate frame to camera frame.
    """
    # Drone rotation (body -> world)
    R_drone_world = R.from_quat(drone_quat_xyzw).as_matrix()

    # Camera rotation (camera -> world)
    R_body_to_cam = R_cam_to_body.T
    R_cam_world = R_drone_world @ R_body_to_cam  # camera frame -> world frame

    # Camera position (assuming camera coincides with drone origin)
    p_cam_world = drone_pos  # since camera is at drone origin

    # Gate position/orientation in world
    p_gate_world = dataset.gate_pos_world
    R_gate_world = dataset.R_gate_world  # gate -> world

    # Gate in camera frame
    t_cam_gate = R_cam_world.T @ (p_gate_world - p_cam_world)
    R_cam_gate = R_cam_world.T @ R_gate_world

    return t_cam_gate, R_cam_gate


def estimate_drone_from_gate_pose(t_cam_gate, R_cam_gate, dataset):
    """
    Given estimated gate pose in camera frame, recover drone pose in world frame.
    Returns: (drone_pos_world, drone_quat_xyzw)
    """
    R_gate_world = dataset.R_gate_world
    p_gate_world = dataset.gate_pos_world

    # R_cam_gate: gate -> camera
    # R_world_cam = R_gate_world @ R_cam_gate.T
    R_world_cam = R_gate_world @ R_cam_gate.T
    t_world_cam = p_gate_world - R_world_cam @ t_cam_gate

    drone_pos = t_world_cam
    drone_quat = R.from_matrix(R_world_cam).as_quat()  # (x,y,z,w)
    return drone_pos, drone_quat


# ----------------------------------------------------------------------
# Main processing pipeline
# ----------------------------------------------------------------------
def main():
    # Paths
    data_dir = r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\anim_test-1"
    model_path = "./models/best.pt"
    output_dir = Path("./output/pnp_test_anim1_4")
    imgs_dir = output_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = RacingDataset(data_dir)
    print(f"Dataset loaded: {len(dataset)} frames")

    # Initialize YOLO
    yolo = YOLO(model_path)

    # Initialize QuAdGate detector
    detector = QuAdGate(min_contour_area=50, epsilon_factor=0.02, confidence_threshold=0.3)

    # Initialize PoseEstimator (using camera intrinsics from dataset)
    pose_estimator = PoseEstimator(
        gate_width=dataset.gate_width,
        gate_height=dataset.gate_height,
        camera_matrix=dataset.camera_matrix,
        dist_coeffs=dataset.dist_coeffs,
        image_size=dataset.image_size,
        camera_fov=None,  # not needed since matrix provided
    )

    # Camera-to-body rotation (from original code)
    R_cam_to_body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

    # Initialize EKF
    ekf = ExtendedKalmanFilter(extrinsic_matrix=R_cam_to_body)
    # Set known gate (only one gate, index 0)
    gate_q_xyzw = R.from_matrix(dataset.R_gate_world).as_quat()  # (x,y,z,w)
    ekf.set_known_gates({0: (dataset.gate_pos_world.copy(), gate_q_xyzw)})

    # Prepare storage for results
    results = []
    # For plotting
    plot_data = {
        "frame": [],
        "time": [],
        # Gate pose in camera frame (estimated vs true)
        "gate_pos_est_x": [],
        "gate_pos_est_y": [],
        "gate_pos_est_z": [],
        "gate_pos_true_x": [],
        "gate_pos_true_y": [],
        "gate_pos_true_z": [],
        "gate_quat_est_x": [],
        "gate_quat_est_y": [],
        "gate_quat_est_z": [],
        "gate_quat_est_w": [],
        "gate_quat_true_x": [],
        "gate_quat_true_y": [],
        "gate_quat_true_z": [],
        "gate_quat_true_w": [],
        # Drone pose in world frame (from PnP, true, EKF)
        "drone_pnp_pos_x": [],
        "drone_pnp_pos_y": [],
        "drone_pnp_pos_z": [],
        "drone_true_pos_x": [],
        "drone_true_pos_y": [],
        "drone_true_pos_z": [],
        "drone_pnp_quat_x": [],
        "drone_pnp_quat_y": [],
        "drone_pnp_quat_z": [],
        "drone_pnp_quat_w": [],
        "drone_true_quat_x": [],
        "drone_true_quat_y": [],
        "drone_true_quat_z": [],
        "drone_true_quat_w": [],
        "drone_ekf_pos_x": [],
        "drone_ekf_pos_y": [],
        "drone_ekf_pos_z": [],
        "drone_ekf_quat_x": [],
        "drone_ekf_quat_y": [],
        "drone_ekf_quat_z": [],
        "drone_ekf_quat_w": [],
        # Reprojection errors (each corner x and y)
        "reproj_err_corner0_x": [],
        "reproj_err_corner0_y": [],
        "reproj_err_corner1_x": [],
        "reproj_err_corner1_y": [],
        "reproj_err_corner2_x": [],
        "reproj_err_corner2_y": [],
        "reproj_err_corner3_x": [],
        "reproj_err_corner3_y": [],
    }

    # Initialize EKF state with first frame ground truth (if available)
    first = dataset[0]
    ekf.reset(position=first["pos_world"].copy(), velocity=np.zeros(3), orientation=first["q_xyzw"].copy())

    prev_time = first["time"]

    # Loop over frames
    for idx in range(len(dataset)):
        print(f"Processing frame {idx:04d} ...")
        data = dataset[idx]
        image_rgb = data["image_rgb"]
        time = data["time"]
        dt = time - prev_time
        prev_time = time

        # ---- YOLO inference ----
        results_yolo = yolo(image_rgb, conf=0.5, iou=0.7, max_det=10, verbose=False)
        result: Results = results_yolo[0]

        # ---- Get mask ----
        if result.masks is not None:
            masks: np.ndarray = (
                result.masks.data if isinstance(result.masks.data, np.ndarray) else result.masks.data.cpu().numpy()
            )
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
            mask = np.max(masks, axis=0)  # merge all masks
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

        # Resize mask to image size (just in case YOLO output differs)
        if mask.shape[:2] != dataset.image_size[::-1]:
            mask = cv2.resize(mask, dataset.image_size, interpolation=cv2.INTER_NEAREST)

        # ---- Gate detection (QuAdGate) ----
        detection = detector.detect(mask)

        # ---- Visualize and save image ----
        # Use result.plot() to get annotated image (BGR)
        vis_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)  # BGR
        # Draw corner annotations if detection exists
        if detection is not None and detection.confidence > 0.3:
            corners = detection.corners.astype(np.int32)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR: TL,TR,BR,BL
            labels = ["TL", "TR", "BR", "BL"]
            for i, (pt, color, label) in enumerate(zip(corners, colors, labels)):
                cv2.circle(vis_img, tuple(pt), 4, color, -1)
                cv2.putText(vis_img, label, (pt[0] + 6, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Draw quadrilateral
            cv2.polylines(vis_img, [corners], True, (0, 255, 255), 2)
            # Confidence
            cv2.putText(
                vis_img, f"Conf: {detection.confidence:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        # Save
        save_path = imgs_dir / f"{idx:04d}.jpg"
        cv2.imwrite(str(save_path), vis_img)

        # ---- Pose estimation (PnP) ----
        gate_pose_est: Optional[GatePose] = None
        if detection is not None and detection.confidence > 0.3:
            gate_pose_est = pose_estimator.estimate_pose(detection, use_ransac=True)

        # ---- Ground truth gate pose in camera frame ----
        true_pos = data["pos_world"]
        true_q_xyzw = data["q_xyzw"]
        t_cam_gate_true, R_cam_gate_true = compute_gate_pose_in_camera(dataset, true_pos, true_q_xyzw, R_cam_to_body)
        q_cam_gate_true = R.from_matrix(R_cam_gate_true).as_quat()  # (x,y,z,w)

        # ---- If PnP succeeded, compute estimates and errors ----
        if gate_pose_est is not None and detection is not None:
            t_cam_gate_est = gate_pose_est.position  # (x,y,z) in camera frame
            R_cam_gate_est = gate_pose_est.rotation_matrix  # gate -> camera
            q_cam_gate_est = gate_pose_est.orientation  # (x,y,z,w)

            # Recover drone pose from PnP
            drone_pos_pnp, drone_q_pnp = estimate_drone_from_gate_pose(t_cam_gate_est, R_cam_gate_est, dataset)

            # Reprojection error per corner
            # image_points = detection.corners.astype(np.float64)
            # obj_points = pose_estimator.gate_points_3d  # (4,3)
            # rvec = gate_pose_est.rvec.reshape(3, 1)
            # tvec = gate_pose_est.tvec.reshape(3, 1)
            # projected, _ = cv2.projectPoints(obj_points, rvec, tvec, dataset.camera_matrix, dataset.dist_coeffs)
            # projected = projected.reshape(-1, 2)
            # reproj_errors = projected - image_points  # (4,2)
            reproj_errors = gate_pose_est.reprojection_errors
            
            # Store for each corner
            for i in range(4):
                plot_data[f"reproj_err_corner{i}_x"].append(reproj_errors[i, 0])
                plot_data[f"reproj_err_corner{i}_y"].append(reproj_errors[i, 1])
        else:
            # No detection: set all estimates to NaN
            t_cam_gate_est = np.full(3, np.nan)
            q_cam_gate_est = np.full(4, np.nan)
            drone_pos_pnp = np.full(3, np.nan)
            drone_q_pnp = np.full(4, np.nan)
            for i in range(4):
                plot_data[f"reproj_err_corner{i}_x"].append(np.nan)
                plot_data[f"reproj_err_corner{i}_y"].append(np.nan)

        # ---- EKF prediction ----
        ekf.predict(dt, data["accel"], data["angvel"])

        # ---- EKF update with vision (if detection valid) ----
        if gate_pose_est is not None and detection is not None and detection.confidence > 0.3:
            ekf.update_gate_pose(t_cam_gate_est, q_cam_gate_est, gate_idx=0)

        # ---- Get EKF state ----
        ekf_state = ekf.state
        drone_pos_ekf = ekf_state.position
        drone_q_ekf = ekf_state.orientation  # (w,x,y,z)? Wait, EKFState stores orientation as (w,x,y,z)? Let's check.
        # In ekf.py, EKFState.orientation is initialized as np.array([1,0,0,0]) which is (w,x,y,z).
        # In as_vector, orientation is placed as 4 elements. So we need to convert to (x,y,z,w) for consistency.
        # However, in the code, orientation is used as (w,x,y,z). In our plotting we want (x,y,z,w).
        # We'll store both but keep (x,y,z,w) for consistency.
        # Let's convert:
        q_ekf_wxyz = drone_q_ekf  # (w,x,y,z)
        q_ekf_xyzw = np.array([q_ekf_wxyz[1], q_ekf_wxyz[2], q_ekf_wxyz[3], q_ekf_wxyz[0]])

        # ---- Store results ----
        frame = idx
        plot_data["frame"].append(frame)
        plot_data["time"].append(time)

        # Gate pose (camera frame)
        plot_data["gate_pos_est_x"].append(t_cam_gate_est[0])
        plot_data["gate_pos_est_y"].append(t_cam_gate_est[1])
        plot_data["gate_pos_est_z"].append(t_cam_gate_est[2])
        plot_data["gate_pos_true_x"].append(t_cam_gate_true[0])
        plot_data["gate_pos_true_y"].append(t_cam_gate_true[1])
        plot_data["gate_pos_true_z"].append(t_cam_gate_true[2])
        if gate_pose_est is not None:
            q_est = q_cam_gate_est
        else:
            q_est = np.full(4, np.nan)
        plot_data["gate_quat_est_x"].append(q_est[0])
        plot_data["gate_quat_est_y"].append(q_est[1])
        plot_data["gate_quat_est_z"].append(q_est[2])
        plot_data["gate_quat_est_w"].append(q_est[3])
        plot_data["gate_quat_true_x"].append(q_cam_gate_true[0])
        plot_data["gate_quat_true_y"].append(q_cam_gate_true[1])
        plot_data["gate_quat_true_z"].append(q_cam_gate_true[2])
        plot_data["gate_quat_true_w"].append(q_cam_gate_true[3])

        # Drone pose from PnP
        plot_data["drone_pnp_pos_x"].append(drone_pos_pnp[0])
        plot_data["drone_pnp_pos_y"].append(drone_pos_pnp[1])
        plot_data["drone_pnp_pos_z"].append(drone_pos_pnp[2])
        plot_data["drone_true_pos_x"].append(true_pos[0])
        plot_data["drone_true_pos_y"].append(true_pos[1])
        plot_data["drone_true_pos_z"].append(true_pos[2])
        if gate_pose_est is not None:
            q_pnp = drone_q_pnp
        else:
            q_pnp = np.full(4, np.nan)
        plot_data["drone_pnp_quat_x"].append(q_pnp[0])
        plot_data["drone_pnp_quat_y"].append(q_pnp[1])
        plot_data["drone_pnp_quat_z"].append(q_pnp[2])
        plot_data["drone_pnp_quat_w"].append(q_pnp[3])
        # True drone quat (already (x,y,z,w))
        plot_data["drone_true_quat_x"].append(true_q_xyzw[0])
        plot_data["drone_true_quat_y"].append(true_q_xyzw[1])
        plot_data["drone_true_quat_z"].append(true_q_xyzw[2])
        plot_data["drone_true_quat_w"].append(true_q_xyzw[3])

        # Drone EKF
        plot_data["drone_ekf_pos_x"].append(drone_pos_ekf[0])
        plot_data["drone_ekf_pos_y"].append(drone_pos_ekf[1])
        plot_data["drone_ekf_pos_z"].append(drone_pos_ekf[2])
        plot_data["drone_ekf_quat_x"].append(q_ekf_xyzw[0])
        plot_data["drone_ekf_quat_y"].append(q_ekf_xyzw[1])
        plot_data["drone_ekf_quat_z"].append(q_ekf_xyzw[2])
        plot_data["drone_ekf_quat_w"].append(q_ekf_xyzw[3])

    # ---- Save results.csv ----
    df_results = pd.DataFrame(plot_data)
    csv_path = output_dir / "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # ---- Generate plots ----
    # Convert quaternions to Euler angles (degrees) for easier visualization
    euler_data = {
        'frame': plot_data['frame'],
        'time': plot_data['time'],
    }

    # For gate pose (camera frame) - mapping from stored keys to output keys
    gate_mapping = [('gate_quat_est', 'gate_est'), ('gate_quat_true', 'gate_true')]
    for store_prefix, out_prefix in gate_mapping:
        qx = np.array(plot_data[f'{store_prefix}_x'])
        qy = np.array(plot_data[f'{store_prefix}_y'])
        qz = np.array(plot_data[f'{store_prefix}_z'])
        qw = np.array(plot_data[f'{store_prefix}_w'])
        euler = np.full((len(qx), 3), np.nan)
        for i in range(len(qx)):
            if not np.isnan(qx[i]):
                rot = R.from_quat([qx[i], qy[i], qz[i], qw[i]])
                euler[i] = np.degrees(rot.as_euler('xyz'))
        euler_data[f'{out_prefix}_roll'] = euler[:,0].tolist()
        euler_data[f'{out_prefix}_pitch'] = euler[:,1].tolist()
        euler_data[f'{out_prefix}_yaw'] = euler[:,2].tolist()

    # For drone pose (world frame) - PnP, true, EKF
    for prefix in ['drone_pnp', 'drone_true', 'drone_ekf']:
        qx = np.array(plot_data[f'{prefix}_quat_x'])
        qy = np.array(plot_data[f'{prefix}_quat_y'])
        qz = np.array(plot_data[f'{prefix}_quat_z'])
        qw = np.array(plot_data[f'{prefix}_quat_w'])
        euler = np.full((len(qx), 3), np.nan)
        for i in range(len(qx)):
            if not np.isnan(qx[i]):
                rot = R.from_quat([qx[i], qy[i], qz[i], qw[i]])
                euler[i] = np.degrees(rot.as_euler('xyz'))
        euler_data[f'{prefix}_roll'] = euler[:,0].tolist()
        euler_data[f'{prefix}_pitch'] = euler[:,1].tolist()
        euler_data[f'{prefix}_yaw'] = euler[:,2].tolist()

    # ---- Combined plots ----
    def plot_combined_drone(euler_data, plot_data, output_dir):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        # Position X
        ax = axes[0]
        ax.plot(plot_data['frame'], plot_data['drone_pnp_pos_x'], label='PnP')
        ax.plot(plot_data['frame'], plot_data['drone_true_pos_x'], label='True')
        ax.plot(plot_data['frame'], plot_data['drone_ekf_pos_x'], label='EKF')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pos X (m)')
        ax.legend()
        ax.grid(True)
        # Position Y
        ax = axes[1]
        ax.plot(plot_data['frame'], plot_data['drone_pnp_pos_y'], label='PnP')
        ax.plot(plot_data['frame'], plot_data['drone_true_pos_y'], label='True')
        ax.plot(plot_data['frame'], plot_data['drone_ekf_pos_y'], label='EKF')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pos Y (m)')
        ax.legend()
        ax.grid(True)
        # Position Z
        ax = axes[2]
        ax.plot(plot_data['frame'], plot_data['drone_pnp_pos_z'], label='PnP')
        ax.plot(plot_data['frame'], plot_data['drone_true_pos_z'], label='True')
        ax.plot(plot_data['frame'], plot_data['drone_ekf_pos_z'], label='EKF')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pos Z (m)')
        ax.legend()
        ax.grid(True)
        # Roll
        ax = axes[3]
        ax.plot(plot_data['frame'], euler_data['drone_pnp_roll'], label='PnP')
        ax.plot(plot_data['frame'], euler_data['drone_true_roll'], label='True')
        ax.plot(plot_data['frame'], euler_data['drone_ekf_roll'], label='EKF')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Roll (deg)')
        ax.legend()
        ax.grid(True)
        # Pitch
        ax = axes[4]
        ax.plot(plot_data['frame'], euler_data['drone_pnp_pitch'], label='PnP')
        ax.plot(plot_data['frame'], euler_data['drone_true_pitch'], label='True')
        ax.plot(plot_data['frame'], euler_data['drone_ekf_pitch'], label='EKF')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pitch (deg)')
        ax.legend()
        ax.grid(True)
        # Yaw
        ax = axes[5]
        ax.plot(plot_data['frame'], euler_data['drone_pnp_yaw'], label='PnP')
        ax.plot(plot_data['frame'], euler_data['drone_true_yaw'], label='True')
        ax.plot(plot_data['frame'], euler_data['drone_ekf_yaw'], label='EKF')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Yaw (deg)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'drone_combined.jpg', dpi=150)
        plt.close()

    def plot_combined_gate(euler_data, plot_data, output_dir):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        # Position X
        ax = axes[0]
        ax.plot(plot_data['frame'], plot_data['gate_pos_est_x'], label='Est')
        ax.plot(plot_data['frame'], plot_data['gate_pos_true_x'], label='True')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pos X (m)')
        ax.legend()
        ax.grid(True)
        # Position Y
        ax = axes[1]
        ax.plot(plot_data['frame'], plot_data['gate_pos_est_y'], label='Est')
        ax.plot(plot_data['frame'], plot_data['gate_pos_true_y'], label='True')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pos Y (m)')
        ax.legend()
        ax.grid(True)
        # Position Z
        ax = axes[2]
        ax.plot(plot_data['frame'], plot_data['gate_pos_est_z'], label='Est')
        ax.plot(plot_data['frame'], plot_data['gate_pos_true_z'], label='True')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pos Z (m)')
        ax.legend()
        ax.grid(True)
        # Roll
        ax = axes[3]
        ax.plot(plot_data['frame'], euler_data['gate_est_roll'], label='Est')
        ax.plot(plot_data['frame'], euler_data['gate_true_roll'], label='True')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Roll (deg)')
        ax.legend()
        ax.grid(True)
        # Pitch
        ax = axes[4]
        ax.plot(plot_data['frame'], euler_data['gate_est_pitch'], label='Est')
        ax.plot(plot_data['frame'], euler_data['gate_true_pitch'], label='True')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pitch (deg)')
        ax.legend()
        ax.grid(True)
        # Yaw
        ax = axes[5]
        ax.plot(plot_data['frame'], euler_data['gate_est_yaw'], label='Est')
        ax.plot(plot_data['frame'], euler_data['gate_true_yaw'], label='True')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Yaw (deg)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'gate_combined.jpg', dpi=150)
        plt.close()

    def plot_combined_reproj(plot_data, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i in range(4):
            ax = axes[i]
            ax.plot(plot_data['frame'], plot_data[f'reproj_err_corner{i}_x'], label='dx')
            ax.plot(plot_data['frame'], plot_data[f'reproj_err_corner{i}_y'], label='dy')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Error (pixels)')
            ax.set_title(f'Corner {i}')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'reproj_combined.jpg', dpi=150)
        plt.close()

    # Generate combined figures
    plot_combined_drone(euler_data, plot_data, output_dir)
    plot_combined_gate(euler_data, plot_data, output_dir)
    plot_combined_reproj(plot_data, output_dir)

    print("All combined plots saved to", output_dir)


if __name__ == "__main__":
    main()
