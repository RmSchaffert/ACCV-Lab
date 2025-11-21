# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pyquaternion import Quaternion


def get_ego_to_world_transform(ego_pose: dict) -> np.ndarray:
    '''Get the ego-to-world transformation matrix for a given ego pose.

    Args:
        ego_pose: Ego pose.

    Returns:
        The resulting transformation matrix.
    '''

    rotation = ego_pose["rotation"]
    translation = ego_pose["translation"]
    ego_to_world = _pose_to_transformation_obj_coords_to_containing_coords(rotation, translation)
    return ego_to_world


def get_sensor_to_ego_transformation(calibrated_sensor: dict) -> np.ndarray:
    '''Get the sensor-to-ego transformation matrix for a given calibrated sensor.

    Args:
        calibrated_sensor: Calibrated sensor.

    Returns:
        The resulting transformation matrix.
    '''

    rotation = calibrated_sensor["rotation"]
    translation = calibrated_sensor["translation"]
    sensor_to_ego = _pose_to_transformation_obj_coords_to_containing_coords(rotation, translation)
    return sensor_to_ego


def get_cam_intrinsics(calibrated_cam: dict) -> np.ndarray:
    '''Get the camera intrinsics for a given calibrated camera.

    Args:
        calibrated_cam: Calibrated camera.

    Returns:
        The resulting camera intrinsics.
    '''
    res = np.array(calibrated_cam['camera_intrinsic'])
    return res


def get_transformation_from_first_to_second(
    calibrated_sensor_from: dict,
    ego_pose_sensor_from: dict,
    calibrated_sensor_to: dict,
    ego_pose_sensor_to: dict,
) -> np.ndarray:
    '''Get the transformation from the first sensor to the second sensor.

    Note:
        Apart from the transformation between the two sensors,
        different ego-poses (due to slighlty different timestamps)
        are also considered.

    The following transformation steps are combined into a single transformation matrix:
    1. [sensor_1] --(sensor_1_to_ego_1)--> [ego_1]
    2. [ego_1] --(ego_1_to_world)--> [world]
    3. [world] --(inv(ego_2_to_world))--> [ego_2]
    4. [ego_2] --(inv(sensor_2_to_ego_2))--> [sensor_2]

    Args:
        calibrated_sensor_from: Calibrated sensor from.
        ego_pose_sensor_from: Ego pose sensor from.
        calibrated_sensor_to: Calibrated sensor to.
        ego_pose_sensor_to: Ego pose sensor to.

    Returns:
        The resulting transformation matrix.
    '''

    sensor_1_to_ego_1 = get_sensor_to_ego_transformation(calibrated_sensor_from)
    ego_1_to_world = get_ego_to_world_transform(ego_pose_sensor_from)
    sensor_2_to_ego_2 = get_sensor_to_ego_transformation(calibrated_sensor_to)
    ego_2_to_world = get_ego_to_world_transform(ego_pose_sensor_to)

    res = (
        np.linalg.inv(sensor_2_to_ego_2) @ np.linalg.inv(ego_2_to_world) @ ego_1_to_world @ sensor_1_to_ego_1
    )
    return res


def _pose_to_transformation_obj_coords_to_containing_coords(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    '''Get the transformation matrix from the object coordinates to the containing coordinates.

    The transformation matrix is a 4x4 matrix which transforms a point from the object coordinates to the containing coordinates.
    The containing coordinates is the reference frame in which the object pose is defined (e.g. the ego-vehicle).

    Args:
        rotation: Rotation quaternion describing the rotation of the object.
        translation: Translation of the object.

    Returns:
        The resulting transformation matrix.
    '''

    res = np.eye(4)
    rotation_mat = Quaternion(rotation).rotation_matrix
    res[0:3, 0:3] = rotation_mat
    res[0:3, 3] = translation
    return res
