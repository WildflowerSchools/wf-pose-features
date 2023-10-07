import pandas as pd
import numpy as np

def impute_keypoints_pose_track(
    pose_track_data,
    timestamp_column_name='timestamp',
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
):
    num_poses = len(pose_track_data)
    pose_track_data_imputed = (
        pose_track_data
        .copy()
        .sort_values('timestamp')
    )
    keypoint_coordinates_flattened = pd.DataFrame(
        np.stack(pose_track_data_imputed[keypoint_coordinates_3d_column_name]).reshape((num_poses, -1)),
        index=pose_track_data_imputed[timestamp_column_name]
    )
    keypoint_coordinates_flattened_interpolated = keypoint_coordinates_flattened.interpolate(method='time')
    pose_track_data_imputed[keypoint_coordinates_3d_column_name] = list(
        np.stack(keypoint_coordinates_flattened_interpolated.values)
        .reshape((num_poses, -1, 3))
    )
    return pose_track_data_imputed

def remove_incomplete_poses(
    pose_data,
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
):
    pose_data_cleaned = (
        pose_data
        .loc[pose_data[keypoint_coordinates_3d_column_name].apply(lambda x: np.all(np.isfinite(x)))]
        .copy()
    )
    return pose_data_cleaned

def generate_shoulders_centers_feature(
    poses,
    left_shoulder_index,
    right_shoulder_index,
):
    shoulders_centers_feature = list()
    for pose in poses:
        shoulders_centers_feature.append(compute_shoulders_center(
            pose=pose,
            left_shoulder_index=left_shoulder_index,
            right_shoulder_index=right_shoulder_index,
        ))
    return shoulders_centers_feature

def compute_shoulders_center(
    pose,
    left_shoulder_index,
    right_shoulder_index,
):
    shoulders_center = np.mean(
        pose[[left_shoulder_index, right_shoulder_index], :],
        axis=0
    )
    return shoulders_center

def generate_poses_recentered_feature(
    poses,
    shoulders_centers,
):
    if len(shoulders_centers) != len(poses):
        raise ValueError(f"Pose object has length {len(poses)} poses but shoulders centers object has length {len(shoulders_center)}")
    poses_recentered_feature = list()
    for pose, shoulders_center in zip(poses, shoulders_centers):
        poses_recentered_feature.append(compute_pose_recentered(
            pose=pose,
            shoulders_center=shoulders_center,
        ))
    return poses_recentered_feature

def compute_pose_recentered(
    pose,
    shoulders_center,
):
    pose_recentered = pose - np.array([[shoulders_center[0],shoulders_center[1], 0.0]])
    return pose_recentered

def generate_shoulder_orientations_feature(
    poses,
    right_shoulder_index,
):
    shoulder_orientations_feature = list()
    for pose in poses:
        shoulder_orientations_feature.append(compute_shoulder_orientation(
            pose=pose,
            right_shoulder_index=right_shoulder_index,
        ))
    return shoulder_orientations_feature

def compute_shoulder_orientation(
    pose,
    right_shoulder_index,
):
    shoulder_orientation = np.arctan2(
        pose[right_shoulder_index, 1],
        pose[right_shoulder_index, 0]
    )
    return shoulder_orientation

def generate_poses_reoriented_feature(
    poses,
    shoulder_orientations,
):
    if len(shoulder_orientations) != len(poses):
        raise ValueError(f"Pose object has length {len(poses)} poses but shoulders orientations object has length {len(shoulder_orientations)}")
    poses_reoriented_feature = list()
    for pose, shoulder_orientation in zip (poses, shoulder_orientations):
        poses_reoriented_feature.append(compute_pose_reoriented(
            pose=pose,
            shoulder_orientation=shoulder_orientation,
        ))
    return poses_reoriented_feature

def compute_pose_reoriented(
    pose,
    shoulder_orientation,
):
    pose_reoriented = apply_z_rotation(
        pose,
        angle=-shoulder_orientation,
    )
    return pose_reoriented

def apply_z_rotation(
    pose,
    angle,
):
    z_rotation_matrix = generate_z_rotation_matrix(angle)
    rotated_pose = np.matmul(pose, z_rotation_matrix.T)
    return rotated_pose

def generate_z_rotation_matrix(angle):
    z_rotation_matrix =  np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return z_rotation_matrix