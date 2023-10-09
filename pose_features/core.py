import pandas as pd
import numpy as np
import scipy
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

def generate_shoulders_xy_feature(
    pose_list,
    keypoint_descriptions,
):
    shoulders_xy = compute_keypoints_xy(
        poses=np.stack(pose_list),
        selected_keypoint_descriptions=['Left shoulder', 'Right shoulder'],
        keypoint_descriptions=keypoint_descriptions,    
    )
    shoulders_xy_list = list(shoulders_xy)
    return shoulders_xy_list

def generate_pose_orientation_xy_shoulders_feature(
    pose_list,
    keypoint_descriptions,
):
    pose_orientations_xy_shoulders = compute_pose_orientations_xy(
        poses=np.stack(pose_list),
        selected_keypoint_descriptions=['Left shoulder', 'Right shoulder'],
        keypoint_descriptions=keypoint_descriptions,
    )
    pose_orientations_xy_shoulders_list = list(pose_orientations_xy_shoulders)
    return pose_orientations_xy_shoulders_list

def poses_body_frame_shoulders_feature(
    pose_list,
    keypoint_descriptions,
):
    poses_body_frame_shoulders = compute_poses_body_frame(
        poses=np.stack(pose_list),
        selected_keypoint_descriptions=['Left shoulder', 'Right shoulder'],
        keypoint_descriptions=keypoint_descriptions,        
    )
    poses_body_frame_shoulders_list = list(poses_body_frame_shoulders)
    return poses_body_frame_shoulders_list

def generate_unit_vector_feature(
    pose_list,
    selected_keypoint_descriptions_from,
    selected_keypoint_descriptions_to,
    keypoint_descriptions,
):
    poses = np.stack(pose_list)
    unit_vectors = compute_unit_vectors(
        poses=poses,
        selected_keypoint_descriptions_from=selected_keypoint_descriptions_from,
        selected_keypoint_descriptions_to=selected_keypoint_descriptions_to,
        keypoint_descriptions=keypoint_descriptions,
    )
    unit_vector_list = list(unit_vectors)
    return unit_vector_list

def compute_unit_vectors(
    poses,
    selected_keypoint_descriptions_from,
    selected_keypoint_descriptions_to,
    keypoint_descriptions,
):
    keypoints_from = compute_keypoints(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions_from,
        keypoint_descriptions=keypoint_descriptions,
    )
    keypoints_to = compute_keypoints(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions_to,
        keypoint_descriptions=keypoint_descriptions,
    )
    unit_vectors = compute_unit_vectors_from_keypoints(
        keypoints_from=keypoints_from,
        keypoints_to=keypoints_to,
    )
    return unit_vectors

def compute_unit_vectors_from_keypoints(
    keypoints_from,
    keypoints_to,
):
    unit_vectors = normalize_vectors(compute_vectors_from_keypoints(
        keypoints_from,
        keypoints_to,
    ))
    return unit_vectors

def compute_vectors_from_keypoints(
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

def compute_poses_body_frame(
    poses,
    selected_keypoint_descriptions,
    keypoint_descriptions,        
):
    if len(selected_keypoint_descriptions) !=2:
        raise ValueError('Two keypoints must be specified to compute poses in the body frame')
    keypoints_xy = compute_keypoints_xy(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions,
        keypoint_descriptions=keypoint_descriptions,    
    )
    pose_orientations_xy = compute_pose_orientations_xy(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions,
        keypoint_descriptions=keypoint_descriptions,
    )
    poses_recentered = np.subtract(
        poses,
        np.expand_dims(
            np.concatenate(
                (
                    keypoints_xy,
                    np.zeros((keypoints_xy.shape[0], 1)),
                ),
                axis=1
            ),
            axis=1
        )
    )
    poses_body_frame = apply_z_rotations(
        poses=poses_recentered,
        angles=-pose_orientations_xy
    )
    return poses_body_frame

def apply_z_rotations(
    poses,
    angles,
):
    z_rotation_matrices = generate_z_rotation_matrices(angles)
    rotated_poses = np.zeros_like(poses)
    num_poses = poses.shape[0]
    for pose_index in range(num_poses):
        rotated_poses[pose_index] = np.matmul(
            poses[pose_index],
            z_rotation_matrices[pose_index].T
        )
    return rotated_poses

def apply_z_rotation(
    pose,
    angle,
):
    z_rotation_matrix = generate_z_rotation_matrix(angle)
    rotated_pose = np.matmul(pose, z_rotation_matrix.T)
    return rotated_pose

def generate_z_rotation_matrices(angles):
    z_rotation_matrices = (
        scipy.spatial.transform.Rotation.from_euler('Z', angles, degrees=False)
        .as_matrix()
    )
    return z_rotation_matrices

def generate_z_rotation_matrix(angle):
    z_rotation_matrix =  np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return z_rotation_matrix

def compute_pose_orientations_xy(
    poses,
    selected_keypoint_descriptions,
    keypoint_descriptions,
):
    if len(selected_keypoint_descriptions) !=2:
        raise ValueError('Two keypoints must be specified to compute X-Y pose orientations')
    midpoints = compute_keypoints(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions,
        keypoint_descriptions=keypoint_descriptions,    
    )
    reference_keypoints = compute_keypoints(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions[1:],
        keypoint_descriptions=keypoint_descriptions,    
    )
    offsets = np.subtract(
        reference_keypoints,
        midpoints,
    )
    pose_orientations_xy = np.arctan2(offsets[:, 1], offsets[:, 0])
    return pose_orientations_xy

def compute_keypoints_xy(
    poses,
    selected_keypoint_descriptions,
    keypoint_descriptions,    
):
    keypoints = compute_keypoints(
        poses=poses,
        selected_keypoint_descriptions=selected_keypoint_descriptions,
        keypoint_descriptions=keypoint_descriptions,
    )
    keypoints_xy = extract_keypoints_xy(
        keypoints
    )
    return keypoints_xy

def extract_keypoints_xy(
    keypoints
):
    keypoints_xy = keypoints[:,:2]
    return keypoints_xy

def compute_keypoints(
    poses,
    selected_keypoint_descriptions,
    keypoint_descriptions,
):
    if len(selected_keypoint_descriptions) == 1:
        keypoints = extract_keypoints(
            poses=poses,
            keypoint_description=selected_keypoint_descriptions[0],
            keypoint_descriptions=keypoint_descriptions
        )
    elif len(selected_keypoint_descriptions) == 2:
        keypoints = compute_midpoints(
            poses=poses,
            keypoint_description_a=selected_keypoint_descriptions[0],
            keypoint_description_b=selected_keypoint_descriptions[1],
            keypoint_descriptions=keypoint_descriptions,
        )
    else:
        raise ValueError('Selected keypoint descriptions must be of length 1 (extract keypoints) or 2 (compute midpoints)')
    return keypoints

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