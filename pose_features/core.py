import pandas as pd
import numpy as np

def impute_keypoints_pose_track(
    pose_track,
    timestamp_column_name='timestamp',
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
):
    num_poses = len(pose_track)
    pose_track_imputed = (
        pose_track
        .copy()
        .sort_values('timestamp')
    )
    keypoint_coordinates_flattened = pd.DataFrame(
        np.stack(pose_track_imputed[keypoint_coordinates_3d_column_name]).reshape((num_poses, -1)),
        index=pose_track_imputed[timestamp_column_name]
    )
    keypoint_coordinates_flattened_interpolated = keypoint_coordinates_flattened.interpolate(method='time')
    pose_track_imputed[keypoint_coordinates_3d_column_name] = list(
        np.stack(keypoint_coordinates_flattened_interpolated.values)
        .reshape((num_poses, -1, 3))
    )
    return pose_track_imputed

def remove_incomplete_poses(
    poses,
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
):
    poses_cleaned = (
        poses
        .loc[poses[keypoint_coordinates_3d_column_name].apply(lambda x: np.all(np.isfinite(x)))]
        .copy()
    )
    return poses_cleaned

def generate_shoulders_center_feature(
    poses,
    left_shoulder_index,
    right_shoulder_index,
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
):
    shoulders_center_feature = poses[keypoint_coordinates_3d_column_name].apply(
        lambda pose: compute_shoulders_center(
            pose=pose,
            left_shoulder_index=left_shoulder_index,
            right_shoulder_index=right_shoulder_index,
        )
    )
    return shoulders_center_feature

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

def generate_pose_recentered_feature(
    poses,
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
    shoulders_center_column_name='shoulders_center'
):
    pose_recentered_feature = poses.apply(
        lambda row: compute_pose_recentered(
            pose=row[keypoint_coordinates_3d_column_name],
            shoulders_center=row[shoulders_center_column_name],
        ),
        axis=1
    )
    return pose_recentered_feature

def compute_pose_recentered(
    pose,
    shoulders_center,
):
    pose_recentered = pose - np.array([[shoulders_center[0],shoulders_center[1], 0.0]])
    return pose_recentered