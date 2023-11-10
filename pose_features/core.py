import pandas as pd
import numpy as np
import scipy
import itertools
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

def generate_pose_pair_data(pose_data):
    pose_track_ids = pose_data['pose_track_3d_id'].unique().tolist()
    pose_track_pairs=list()
    for pose_track_3d_id_a, pose_track_3d_id_b in itertools.combinations(pose_track_ids, 2):
        pose_track_a = (
            pose_data
            .loc[pose_data['pose_track_3d_id'] == pose_track_3d_id_a]
            .reset_index()
            .set_index('timestamp')
        )
        pose_track_b = (
            pose_data
            .loc[pose_data['pose_track_3d_id'] == pose_track_3d_id_b]
            .reset_index()
            .set_index('timestamp')
        )
        pose_track_pair = (
            pose_track_a
            .join(
                pose_track_b,
                how='inner',
                lsuffix='_a',
                rsuffix='_b'
            )
            .reset_index()
            .set_index([
                'pose_3d_id_a',
                'pose_3d_id_b',
            ])
        )
        if len(pose_track_pair) > 0:
            pose_track_pairs.append(pose_track_pair)
    pose_pair_data = pd.concat(pose_track_pairs)
    return pose_pair_data

def generate_poses_body_frame_feature(
    pose_series,
    pose_track_id_series,
    selected_keypoint_names,
    keypoint_names,
):
    poses_body_frame_series = (
        pose_series
        .groupby(pose_track_id_series, group_keys=False)
        .apply(lambda x: generate_poses_body_frame_feature_pose_track(
            pose_series=x,
            selected_keypoint_names=selected_keypoint_names,
            keypoint_names=keypoint_names,
        ))
    )
    return poses_body_frame_series

def generate_poses_body_frame_feature_pose_track(
    pose_series,
    selected_keypoint_names,
    keypoint_names,
):
    poses_body_frame = compute_poses_body_frame(
        poses=np.stack(pose_series.values),
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,        
    )
    poses_body_frame_series = pd.Series(
        list(poses_body_frame),
        index=pose_series.index
    )
    return poses_body_frame_series

def compute_poses_body_frame(
    poses,
    selected_keypoint_names,
    keypoint_names,        
):
    if len(selected_keypoint_names) !=2:
        raise ValueError('Two keypoints must be specified to compute poses in the body frame')
    keypoints_xy = compute_keypoints_xy(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,    
    )
    pose_orientations_xy = compute_pose_orientations_xy(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,
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

def generate_z_rotation_matrices(angles):
    z_rotation_matrices = (
        scipy.spatial.transform.Rotation.from_euler('Z', angles, degrees=False)
        .as_matrix()
    )
    return z_rotation_matrices

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

def generate_pose_center_xy_feature(
    pose_series,
    selected_keypoint_names,
    keypoint_names,
):
    pose_centers_xy = compute_keypoints_xy(
        poses=np.stack(pose_series.values),
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,    
    )
    pose_centers_xy_series = pd.Series(
        list(pose_centers_xy),
        index=pose_series.index
    )
    return pose_centers_xy_series

def compute_keypoints_xy(
    poses,
    selected_keypoint_names,
    keypoint_names,    
):
    keypoints = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,
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

def generate_pose_orientation_xy_feature(
    pose_series,
    pose_track_id_series,
    selected_keypoint_names,
    keypoint_names,
):
    pose_orientations_xy_series = (
        pose_series
        .groupby(pose_track_id_series, group_keys=False)
        .apply(lambda x: generate_pose_orientation_xy_feature_pose_track(
            pose_series=x,
            selected_keypoint_names=selected_keypoint_names,
            keypoint_names=keypoint_names,
        ))
    )
    return pose_orientations_xy_series

def generate_pose_orientation_xy_feature_pose_track(
    pose_series,
    selected_keypoint_names,
    keypoint_names,
):
    pose_orientations_xy = compute_pose_orientations_xy(
        poses=np.stack(pose_series.values),
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,
    )
    pose_orientations_xy_series = pd.Series(
        list(pose_orientations_xy),
        index=pose_series.index,
    )
    return pose_orientations_xy_series

def compute_pose_orientations_xy(
    poses,
    selected_keypoint_names,
    keypoint_names,
):
    if len(selected_keypoint_names) !=2:
        raise ValueError('Two keypoints must be specified to compute X-Y pose orientations')
    midpoints = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,    
    )
    reference_keypoints = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names[1:],
        keypoint_names=keypoint_names,    
    )
    offsets = np.subtract(
        reference_keypoints,
        midpoints,
    )
    pose_orientations_xy = remove_angle_discontinuities(np.arctan2(offsets[:, 1], offsets[:, 0]))
    return pose_orientations_xy

def generate_vector_angles_spherical_feature(
    pose_series,
    pose_track_id_series,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
):
    vector_angles_series = (
        pose_series
        .groupby(pose_track_id_series, group_keys=False)
        .apply(lambda x: generate_vector_angles_spherical_feature_pose_track(
            pose_series=x,
            selected_keypoint_names_from=selected_keypoint_names_from,
            selected_keypoint_names_to=selected_keypoint_names_to,
            keypoint_names=keypoint_names,
        ))
    )
    return vector_angles_series

def generate_vector_angles_spherical_feature_pose_track(
    pose_series,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
):
    vector_angles = compute_vector_angles_spherical(
        poses=np.stack(pose_series.values),
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    vector_angles_series = pd.Series(
        list(vector_angles),
        index=pose_series.index,
    )
    return vector_angles_series

def compute_vector_angles_spherical(
    poses,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,        
):
    unit_vectors = compute_unit_vectors(
        poses=poses,
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    phis = np.arccos(unit_vectors[:, 2])
    thetas = remove_angle_discontinuities(np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0]))
    vector_angles = np.stack(
        (phis, thetas),
        axis=1
    )
    return vector_angles

def generate_vector_angles_zy_feature(
    pose_series,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
    include_beta=False,
):
    vector_angles = compute_vector_angles_zy(
        poses=np.stack(pose_series.values),
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
        include_beta=include_beta,
    )
    vector_angles_series = pd.Series(
        list(vector_angles),
        index=pose_series.index
    )
    return vector_angles_series

def compute_vector_angles_zy(
    poses,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
    include_beta=False,
):
    unit_vectors = compute_unit_vectors(
        poses=poses,
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    phis = np.arccos(unit_vectors[:, 2])
    alphas = np.arccos(unit_vectors[:, 1]/np.linalg.norm(unit_vectors[:, :2], axis=1))
    if include_beta:
        betas = np.sign(unit_vectors[:, 0])
        vector_angles = np.stack(
            (phis, alphas, betas),
            axis=1
        )
    else:
        vector_angles = np.stack(
            (phis, alphas),
            axis=1
        )
    return vector_angles

def generate_angle_between_vectors_pose_pair_feature(
    pose_series_from_a,
    selected_keypoint_names_from_a,
    pose_series_to_a,
    selected_keypoint_names_to_a,
    pose_series_from_b,
    selected_keypoint_names_from_b,
    pose_series_to_b,
    selected_keypoint_names_to_b,
    keypoint_names,
):
    angles = compute_angle_between_vectors_pose_pair(
        poses_from_a=np.stack(pose_series_from_a.values),
        selected_keypoint_names_from_a=selected_keypoint_names_from_a,
        poses_to_a=np.stack(pose_series_to_a.values),
        selected_keypoint_names_to_a=selected_keypoint_names_to_a,
        poses_from_b=np.stack(pose_series_from_b.values),
        selected_keypoint_names_from_b=selected_keypoint_names_from_b,
        poses_to_b=np.stack(pose_series_to_b.values),
        selected_keypoint_names_to_b=selected_keypoint_names_to_b,
        keypoint_names=keypoint_names,
    )
    angles_series = pd.Series(
        list(angles),
        index=pose_series_from_a.index
    )
    return angles_series

def generate_angle_between_vectors_feature(
    pose_series,
    selected_keypoint_names_from_a,
    selected_keypoint_names_to_a,
    selected_keypoint_names_from_b,
    selected_keypoint_names_to_b,
    keypoint_names,
):
    angles = compute_angle_between_vectors(
        poses=np.stack(pose_series.values),
        selected_keypoint_names_from_a=selected_keypoint_names_from_a,
        selected_keypoint_names_to_a=selected_keypoint_names_to_a,
        selected_keypoint_names_from_b=selected_keypoint_names_from_b,
        selected_keypoint_names_to_b=selected_keypoint_names_to_b,
        keypoint_names=keypoint_names,
    )
    angles_series = pd.Series(
        list(angles),
        index=pose_series.index
    )
    return angles_series

def compute_angle_between_vectors_pose_pair(
    poses_from_a,
    selected_keypoint_names_from_a,
    poses_to_a,
    selected_keypoint_names_to_a,
    poses_from_b,
    selected_keypoint_names_from_b,
    poses_to_b,
    selected_keypoint_names_to_b,
    keypoint_names,        
):
    unit_vectors_a = compute_unit_vectors_pose_pair(
        poses_from=poses_from_a,
        selected_keypoint_names_from=selected_keypoint_names_from_a,
        poses_to=poses_to_a,
        selected_keypoint_names_to=selected_keypoint_names_to_a,
        keypoint_names=keypoint_names,
    )
    unit_vectors_b = compute_unit_vectors_pose_pair(
        poses_from=poses_from_b,
        selected_keypoint_names_from=selected_keypoint_names_from_b,
        poses_to=poses_to_b,
        selected_keypoint_names_to=selected_keypoint_names_to_b,
        keypoint_names=keypoint_names,
    )
    angles = np.arccos(np.sum(
        np.multiply(
            unit_vectors_a,
            unit_vectors_b
        ),
        axis=1
    ))
    return angles

def compute_angle_between_vectors(
    poses,
    selected_keypoint_names_from_a,
    selected_keypoint_names_to_a,
    selected_keypoint_names_from_b,
    selected_keypoint_names_to_b,
    keypoint_names,        
):
    unit_vectors_a = compute_unit_vectors(
        poses=poses,
        selected_keypoint_names_from=selected_keypoint_names_from_a,
        selected_keypoint_names_to=selected_keypoint_names_to_a,
        keypoint_names=keypoint_names,
    )
    unit_vectors_b = compute_unit_vectors(
        poses=poses,
        selected_keypoint_names_from=selected_keypoint_names_from_b,
        selected_keypoint_names_to=selected_keypoint_names_to_b,
        keypoint_names=keypoint_names,
    )
    angles = np.arccos(np.sum(
        np.multiply(
            unit_vectors_a,
            unit_vectors_b
        ),
        axis=1
    ))
    return angles

def generate_unit_vector_pose_pair_feature(
    pose_series_from,
    selected_keypoint_names_from,
    pose_series_to,
    selected_keypoint_names_to,
    keypoint_names,
):
    unit_vectors = compute_unit_vectors_pose_pair(
        poses_from=np.stack(pose_series_from.values),
        selected_keypoint_names_from=selected_keypoint_names_from,
        poses_to=np.stack(pose_series_to.values),
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    unit_vector_series = pd.Series(
        list(unit_vectors),
        index=pose_series_from.index
    )
    return unit_vector_series

def generate_unit_vector_feature(
    pose_series,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
):
    unit_vectors = compute_unit_vectors(
        poses=np.stack(pose_series.values),
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    unit_vector_series = pd.Series(
        list(unit_vectors),
        index=pose_series.index
    )
    return unit_vector_series

def compute_unit_vectors_pose_pair(
    poses_from,
    selected_keypoint_names_from,
    poses_to,
    selected_keypoint_names_to,
    keypoint_names,
):
    keypoints_from = compute_keypoints(
        poses=poses_from,
        selected_keypoint_names=selected_keypoint_names_from,
        keypoint_names=keypoint_names,
    )
    keypoints_to = compute_keypoints(
        poses=poses_to,
        selected_keypoint_names=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    unit_vectors = compute_unit_vectors_from_keypoints(
        keypoints_from=keypoints_from,
        keypoints_to=keypoints_to,
    )
    return unit_vectors

def compute_unit_vectors(
    poses,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
):
    keypoints_from = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names_from,
        keypoint_names=keypoint_names,
    )
    keypoints_to = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
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

def generate_time_derivative_feature(
    object_series,
    timestamp_series,
    pose_track_id_series,
):
    time_series = pd.DataFrame({
        'timestamp': timestamp_series,
        'object': object_series,
    })
    object_derivative_series = (
        time_series
        .groupby(pose_track_id_series, group_keys=False)
        .apply(lambda x: generate_time_derivative_feature_pose_track(
            object_series=x['object'],
            timestamp_series=x['timestamp'],
        ))
    )
    return object_derivative_series

def generate_time_derivative_feature_pose_track(
    object_series,
    timestamp_series,
):
    objects_derivative = compute_time_derivative(
        objects=np.stack(object_series.values),
        timestamps=np.stack(timestamp_series.values),
    )
    object_derivative_series = pd.Series(
        list(objects_derivative),
        index=object_series.index
    )
    return object_derivative_series

def compute_time_derivative(
    objects,
    timestamps,
):
    timestamps_seconds = (timestamps - timestamps[0])/np.timedelta64(1, 's')
    objects_derivative = np.apply_along_axis(lambda x: np.gradient(x, timestamps_seconds), axis=0, arr=objects)
    return objects_derivative

def generate_time_average_feature(
    object_series,
    timestamp_series,
    pose_track_id_series,
    time_window,
):
    time_series = pd.DataFrame({
        'timestamp': timestamp_series,
        'object': object_series,
    })
    object_averaged_series = (
        time_series
        .groupby(pose_track_id_series, group_keys=False)
        .apply(lambda x: generate_time_average_feature_pose_track(
            object_series=x['object'],
            timestamp_series=x['timestamp'],
            time_window=time_window,
        ))
    )
    return object_averaged_series

def generate_time_average_feature_pose_track(
    object_series,
    timestamp_series,
    time_window,
):
    objects_averaged = compute_time_average(
        objects=np.stack(object_series.values),
        timestamps=np.stack(timestamp_series.values),
        time_window=time_window,
    )
    object_averaged_series = pd.Series(
        list(objects_averaged),
        index=object_series.index
    )
    return object_averaged_series

def compute_time_average(
    objects,
    timestamps,
    time_window,
):
    num_objects = objects.shape[0]
    objects_averaged = (
        pd.DataFrame(
            objects.reshape((num_objects, -1)),
            index=timestamps
        )
        .rolling(time_window, center=True)
        .mean()
        .values
        .reshape(objects.shape)
    )
    return objects_averaged

def generate_magnitude_squared_feature(feature_series):
    magnitude_squared_objects = np.sum(np.square(np.stack(feature_series.values)), axis=-1)
    feature_magnitude_squared_series = pd.Series(
        list(magnitude_squared_objects),
        index=feature_series.index
    )
    return feature_magnitude_squared_series

def generate_squared_feature(feature_series):
    squared_objects = np.square(np.stack(feature_series.values))
    feature_squared_series = pd.Series(
        list(squared_objects),
        index=feature_series.index
    )
    return feature_squared_series

def flatten_feature(
    feature_series,
    dimension_names,
):
    feature_name = feature_series.name
    column_names = list()
    for dimension_name_tuple in itertools.product(*dimension_names):
        column_names.append('_'.join([
            feature_name,
            '_'.join(dimension_name_tuple)
        ]))
    feature_flattened = pd.DataFrame(
        np.stack(feature_series.values).reshape((len(feature_series), -1)),
        index=feature_series.index,
        columns=column_names
    )
    return feature_flattened

def compute_keypoints(
    poses,
    selected_keypoint_names,
    keypoint_names,
):
    if len(selected_keypoint_names) == 1:
        keypoints = extract_keypoints(
            poses=poses,
            keypoint_name=selected_keypoint_names[0],
            keypoint_names=keypoint_names
        )
    elif len(selected_keypoint_names) == 2:
        keypoints = compute_midpoints(
            poses=poses,
            keypoint_name_a=selected_keypoint_names[0],
            keypoint_name_b=selected_keypoint_names[1],
            keypoint_names=keypoint_names,
        )
    else:
        raise ValueError('Selected keypoint names must be of length 1 (extract keypoints) or 2 (compute midpoints)')
    return keypoints

def compute_midpoints(
    poses,
    keypoint_name_a,
    keypoint_name_b,
    keypoint_names,
):
    index_a = find_keypoint_index(keypoint_name_a, keypoint_names)
    index_b = find_keypoint_index(keypoint_name_b, keypoint_names)
    midpoints = np.mean(
        poses[:, [index_a, index_b], :],
        axis=1
    )
    return midpoints

def extract_keypoints(
    poses,
    keypoint_name,
    keypoint_names
):
    keypoint_index = find_keypoint_index(
        keypoint_name,
        keypoint_names
    )
    keypoints = poses[:,  keypoint_index]
    return keypoints

def find_keypoint_index(
    keypoint_name,
    keypoint_names
):
    keypoint_index = keypoint_names.index(keypoint_name)
    return keypoint_index

def remove_angle_discontinuities(angles):
    diff = np.concatenate((
        np.array([0.0,]),
        np.diff(angles)
    ))
    diff_without_discontinuities = np.where(
        diff < -np.pi,
        diff + 2*np.pi,
        np.where(
            diff > np.pi,
            diff - 2*np.pi,
            diff
        )
    )
    angles_without_discontinuities = angles[0] + np.cumsum(diff_without_discontinuities)
    return angles_without_discontinuities

def generate_poses_body_frame_feature_old(
    pose_series,
    selected_keypoint_names,
    keypoint_names,
):
    poses_body_frame = compute_poses_body_frame_old(
        poses=np.stack(pose_series.values),
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,        
    )
    poses_body_frame_series = pd.Series(
        list(poses_body_frame),
        pose_series.index
    )
    return poses_body_frame_series

def compute_poses_body_frame_old(
    poses,
    selected_keypoint_names,
    keypoint_names,        
):
    if len(selected_keypoint_names) !=2:
        raise ValueError('Two keypoints must be specified to compute poses in the body frame')
    keypoints_xy = compute_keypoints_xy(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,    
    )
    pose_orientations_xy = compute_pose_orientations_xy_old(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,
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

def generate_pose_orientation_xy_feature_old(
    pose_series,
    selected_keypoint_names,
    keypoint_names,
):
    pose_orientations_xy = compute_pose_orientations_xy_old(
        poses=np.stack(pose_series.values),
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,
    )
    pose_orientations_xy_series = pd.Series(
        list(pose_orientations_xy),
        index=pose_series.index
    )
    return pose_orientations_xy_series

def compute_pose_orientations_xy_old(
    poses,
    selected_keypoint_names,
    keypoint_names,
):
    if len(selected_keypoint_names) !=2:
        raise ValueError('Two keypoints must be specified to compute X-Y pose orientations')
    midpoints = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names,
        keypoint_names=keypoint_names,    
    )
    reference_keypoints = compute_keypoints(
        poses=poses,
        selected_keypoint_names=selected_keypoint_names[1:],
        keypoint_names=keypoint_names,    
    )
    offsets = np.subtract(
        reference_keypoints,
        midpoints,
    )
    pose_orientations_xy = np.arctan2(offsets[:, 1], offsets[:, 0])
    return pose_orientations_xy

def generate_vector_angles_spherical_feature_old(
    pose_series,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,
):
    vector_angles = compute_vector_angles_spherical_old(
        poses=np.stack(pose_series.values),
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    vector_angles_series = pd.Series(
        list(vector_angles),
        index=pose_series.index
    )
    return vector_angles_series

def compute_vector_angles_spherical_old(
    poses,
    selected_keypoint_names_from,
    selected_keypoint_names_to,
    keypoint_names,        
):
    unit_vectors = compute_unit_vectors(
        poses=poses,
        selected_keypoint_names_from=selected_keypoint_names_from,
        selected_keypoint_names_to=selected_keypoint_names_to,
        keypoint_names=keypoint_names,
    )
    phis = np.arccos(unit_vectors[:, 2])
    thetas = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])
    vector_angles = np.stack(
        (phis, thetas),
        axis=1
    )
    return vector_angles



