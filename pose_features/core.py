import pandas as pd
import numpy as np
import functools

def impute_keypoints(
    pose_data,
    pose_track_id_column_name='pose_track_3d_id',
    timestamp_column_name='timestamp',
    keypoint_coordinates_3d_column_name='keypoint_coordinates_3d',
):
    impute_keypoints_pose_track_partial = functools.partial(
        impute_keypoints_pose_track,
        timestamp_column_name=timestamp_column_name,
        keypoint_coordinates_3d_column_name=keypoint_coordinates_3d_column_name,
    )
    pose_data_imputed = (
        pose_data
        .groupby(
            pose_track_id_column_name,
            as_index=False,
            group_keys=False,
        )
        .apply(impute_keypoints_pose_track_partial)
    )
    return pose_data_imputed

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

def generate_neck_head_vector_feature(
    poses,
    keypoint_descriptions,
):
    neck_head_vectors = compute_neck_head_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    neck_head_vector_feature = list(neck_head_vectors)
    return neck_head_vector_feature

def compute_neck_head_vectors(
    poses,
    keypoint_descriptions,
):
    neck_head_vectors = compute_unit_vectors(
        keypoints_from=extract_keypoints(
            poses=poses,
            keypoint_description='Neck',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=extract_keypoints(
            poses=poses,
            keypoint_description='Head',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return neck_head_vectors

def generate_ears_nose_vector_feature(
    poses,
    keypoint_descriptions,
):
    ears_nose_vectors = compute_ears_nose_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    ears_nose_vector_feature = list(ears_nose_vectors)
    return ears_nose_vector_feature

def compute_ears_nose_vectors(
    poses,
    keypoint_descriptions,
):
    ears_nose_vectors = compute_unit_vectors(
        keypoints_from=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left ear',
            keypoint_description_b='Right ear',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=extract_keypoints(
            poses=poses,
            keypoint_description='Nose',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return ears_nose_vectors

def generate_ears_eyes_vector_feature(
    poses,
    keypoint_descriptions,
):
    ears_eyes_vectors = compute_ears_eyes_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    ears_eyes_vector_feature = list(ears_eyes_vectors)
    return ears_eyes_vector_feature

def compute_ears_eyes_vectors(
    poses,
    keypoint_descriptions,
):
    ears_eyes_vectors = compute_unit_vectors(
        keypoints_from=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left ear',
            keypoint_description_b='Right ear',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left eye',
            keypoint_description_b='Right eye',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return ears_eyes_vectors

def generate_ears_left_wrist_vector_feature(
    poses,
    keypoint_descriptions,
):
    ears_left_wrist_vectors = compute_ears_left_wrist_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    ears_left_wrist_vector_feature = list(ears_left_wrist_vectors)
    return ears_left_wrist_vector_feature

def compute_ears_left_wrist_vectors(
    poses,
    keypoint_descriptions,
):
    ears_left_wrist_vectors = compute_unit_vectors(
        keypoints_from=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left ear',
            keypoint_description_b='Right ear',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=extract_keypoints(
            poses=poses,
            keypoint_description='Left wrist',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return ears_left_wrist_vectors

def generate_ears_right_wrist_vector_feature(
    poses,
    keypoint_descriptions,
):
    ears_right_wrist_vectors = compute_ears_right_wrist_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    ears_right_wrist_vector_feature = list(ears_right_wrist_vectors)
    return ears_right_wrist_vector_feature

def compute_ears_right_wrist_vectors(
    poses,
    keypoint_descriptions,
):
    ears_right_wrist_vectors = compute_unit_vectors(
        keypoints_from=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left ear',
            keypoint_description_b='Right ear',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=extract_keypoints(
            poses=poses,
            keypoint_description='Right wrist',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return ears_right_wrist_vectors

def generate_shoulders_left_wrist_vector_feature(
    poses,
    keypoint_descriptions,
):
    shoulders_left_wrist_vectors = compute_shoulders_left_wrist_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    shoulders_left_wrist_vector_feature = list(shoulders_left_wrist_vectors)
    return shoulders_left_wrist_vector_feature

def compute_shoulders_left_wrist_vectors(
    poses,
    keypoint_descriptions,
):
    shoulders_left_wrist_vectors = compute_unit_vectors(
        keypoints_from=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left shoulder',
            keypoint_description_b='Right shoulder',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=extract_keypoints(
            poses=poses,
            keypoint_description='Left wrist',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return shoulders_left_wrist_vectors

def generate_shoulders_right_wrist_vector_feature(
    poses,
    keypoint_descriptions,
):
    shoulders_right_wrist_vectors = compute_shoulders_right_wrist_vectors(
        poses=np.stack(poses),
        keypoint_descriptions=keypoint_descriptions,
    )
    shoulders_right_wrist_vector_feature = list(shoulders_right_wrist_vectors)
    return shoulders_right_wrist_vector_feature

def compute_shoulders_right_wrist_vectors(
    poses,
    keypoint_descriptions,
):
    shoulders_right_wrist_vectors = compute_unit_vectors(
        keypoints_from=compute_midpoints(
            poses=poses,
            keypoint_description_a='Left shoulder',
            keypoint_description_b='Right shoulder',
            keypoint_descriptions=keypoint_descriptions,
        ),
        keypoints_to=extract_keypoints(
            poses=poses,
            keypoint_description='Right wrist',
            keypoint_descriptions=keypoint_descriptions,
        ),
    )
    return shoulders_right_wrist_vectors

def compute_unit_vectors(
    keypoints_from,
    keypoints_to,
):
    unit_vectors = normalize_vectors(compute_vectors(
        keypoints_from,
        keypoints_to,
    ))
    return unit_vectors

def compute_vectors(
    keypoints_from,
    keypoints_to,
):
    vectors = np.subtract(
        keypoints_to,
        keypoints_from,
    )
    return vectors

def normalize_vectors(vectors):
    normalized_vector = np.divide(
        vectors,
        np.linalg.norm(vectors, axis=1, keepdims=True)
    )
    return normalized_vector

def compute_midpoints(
    poses,
    keypoint_description_a,
    keypoint_description_b,
    keypoint_descriptions,
):
    index_a = find_keypoint_index(keypoint_description_a, keypoint_descriptions)
    index_b = find_keypoint_index(keypoint_description_b, keypoint_descriptions)
    midpoints = np.mean(
        poses[:, [index_a, index_b], :],
        axis=1
    )
    return midpoints

def extract_keypoints(
    poses,
    keypoint_description,
    keypoint_descriptions
):
    keypoint_index = find_keypoint_index(
        keypoint_description,
        keypoint_descriptions
    )
    keypoints = poses[:,  keypoint_index]
    return keypoints

def find_keypoint_index(
    keypoint_description,
    keypoint_descriptions
):
    keypoint_index = keypoint_descriptions.index(keypoint_description)
    return keypoint_index