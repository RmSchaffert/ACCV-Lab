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

from typing import List

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

import nvidia.dali.types as types

from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup
from accvlab.dali_pipeline_framework.inputs import DataProvider

from .geometry_helpers import (
    get_ego_to_world_transform,
    get_sensor_to_ego_transformation,
    get_cam_intrinsics,
    get_transformation_from_first_to_second,
)

from .nuscenes_data_converter import NuScenesDataConverter
from .nuscenes_reader import NuScenesReader
from .nuscenes_data import NuScenesData, NuScenesDataSample


class NuscenesStreamPETRDataProvider(DataProvider):
    '''Data provider for the StreamPETR task using NuScenes data.

    Can be used with the available input callables/iterables.
    '''

    _num_cams = 6

    _mapping_categories_3d = {
        "vehicle.car": 0,
        "vehicle.truck": 1,
        "vehicle.construction": 2,
        "vehicle.bus": 3,
        "vehicle.trailer": 4,
        "movable_object.barrier": 5,
        "vehicle.motorcycle": 6,
        "vehicle.bicycle": 7,
        "human.pedestrian": 8,
        "movable_object.trafficcone": 9,
        None: -1,
    }
    _mapping_caterogies_2d = {
        "vehicle.car": 0,
        "vehicle.truck": 1,
        "vehicle.trailer": 2,
        "vehicle.bus": 3,
        "vehicle.construction": 4,
        "vehicle.bicycle": 5,
        "vehicle.motorcycle": 6,
        "human.pedestrian": 7,
        "movable_object.trafficcone": 8,
        "movable_object.barrier": 9,
        None: -1,
    }

    _mapping_visibility = {'v0-40': 0, 'v40-60': 1, 'v60-80': 3, 'v80-100': 4}

    def __init__(
        self,
        base_dir: str,
        data: NuScenesData,
        image_size=[1600, 900],
        return_ground_truth: bool = True,
    ):
        '''

        Args:
            base_dir: Base directory of the NuScenes dataset.
            data: Dataset meta-data.
            image_size: Size of the images.
            return_ground_truth: Whether to return the ground truth data.
        '''

        self._data = data
        self._converter = NuScenesDataConverter(base_dir, 2, False)
        self._sample_data_structure = self._prepare_needed_sample_data_structure(
            self._num_cams, return_ground_truth
        )
        self._categories_3d_set = set(self._mapping_categories_3d.keys())
        self._categories_3d_set.discard(None)
        self._return_ground_truth = return_ground_truth

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        '''Get the data for a given sample ID.

        Args:
            sample_id: ID of the sample to get the data for.

        Returns:
            Data for the given sample ID.
        '''

        # Get the indices (sequence & sample idx)
        sequence_idx, in_sequence_idx = self._data.get_sequence_and_sample_idx_from_flattened_idx(sample_id)

        # Get training sample (SampleDataGroup) to fill with data
        training_sample = self._sample_data_structure.get_empty_like_self()

        self._prepare_data(training_sample, sequence_idx, in_sequence_idx)

        return training_sample

    @override
    def get_number_of_samples(self) -> int:
        '''Get the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        '''
        data_len = self._data.total_num_samples
        return data_len

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        '''Get the sample data structure (blueprint).'''
        return self._sample_data_structure.get_empty_like_self()

    @staticmethod
    def get_numeric_to_string_categories() -> List[str]:
        '''Get the mapping from numeric to string categories (class labels).

        Note:
            The mapping is a list, where the index corresponds to the numeric category and the value
            corresponds to the string category.

        Returns:
            The mapping from numeric to string categories.
        '''
        res = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]
        return res

    def _prepare_data(
        self,
        training_sample: SampleDataGroup,
        sequence_idx: int,
        in_sequence_idx: int,
    ):
        '''Prepare the data

        Args:
            training_sample: Training sample to fill with data. The data will be filled
                in-place into the sample.
            sequence_idx: Index of the sequence.
            in_sequence_idx: Index of the sample in the sequence.
        '''
        has_prev = in_sequence_idx > 0

        nuscenes_sample = self._data.get_sample_in_sequence(sequence_idx, in_sequence_idx)

        training_sample["prev_exists"] = has_prev

        # --------- Ego Pose ---------
        ego_pose = nuscenes_sample.ego_pose_at_lidar_timestamp

        ego_to_world = get_ego_to_world_transform(ego_pose)
        lidar_to_ego = get_sensor_to_ego_transformation(nuscenes_sample.lidar_calibtation_data)
        lidar_to_world = ego_to_world @ lidar_to_ego

        world_to_lidar = np.linalg.inv(lidar_to_world)

        training_sample["ego_pose_geom"]["lidar_ego_pose"] = lidar_to_world
        training_sample["ego_pose_geom"]["lidar_ego_pose_inv"] = world_to_lidar

        # --------- Sample Timestamp ---------
        timestamp_double = nuscenes_sample.sample["timestamp"] * 1e-6
        training_sample["timestamp"] = timestamp_double

        # --------- Annotation for Images ---------
        if self._return_ground_truth:
            sample_2d_annotations = [
                self._converter.prepare_projected_bboxes_annotations(nuscenes_sample, cam_idx)
                for cam_idx in range(self._num_cams)
            ]

        # --------- Cameras ---------
        for cam_idx in range(self._num_cams):

            training_sample_cam = training_sample["cams"][cam_idx]

            # --------- Image ---------
            image = self._converter.prepare_image_data(nuscenes_sample, cam_idx)
            training_sample_cam["image"] = image

            # --------- Lidar 2 Image Transformation ---------
            lidar_to_cam = get_transformation_from_first_to_second(
                nuscenes_sample.lidar_calibtation_data,
                nuscenes_sample.ego_pose_at_lidar_timestamp,
                nuscenes_sample.camera_calibration[cam_idx],
                nuscenes_sample.ego_poses_at_cam_timestamps[cam_idx],
            )
            camera_intrinsics = get_cam_intrinsics(nuscenes_sample.camera_calibration[cam_idx])
            lidar_to_image = camera_intrinsics @ lidar_to_cam[0:3, :]

            training_sample_cam["cam_geometry"]["lidar2img"] = lidar_to_image
            training_sample_cam["cam_geometry"]["intr_lidar2img"] = camera_intrinsics
            training_sample_cam["cam_geometry"]["extr_lidar2img"] = lidar_to_cam

            # --------- Camera timestamp ---------
            timestamp_double = nuscenes_sample.camera_samples[cam_idx]["timestamp"] * 1e-6
            training_sample_cam["timestamp"] = timestamp_double

            # --------- Image Ground Truth Annotation ---------
            if self._return_ground_truth:
                training_sample_cam_gt_boxes_2d = training_sample_cam["gt_boxes_2d"]
                image_annotation = sample_2d_annotations[cam_idx]
                training_sample_cam_gt_boxes_2d["bboxes"] = image_annotation.bboxes
                training_sample_cam_gt_boxes_2d["categories"] = image_annotation.categories
                training_sample_cam_gt_boxes_2d["depths"] = image_annotation.depths
                training_sample_cam_gt_boxes_2d["centers"] = image_annotation.centers
                training_sample_cam_gt_boxes_2d["num_lidar_points"] = image_annotation.num_lidar_points
                training_sample_cam_gt_boxes_2d["num_radar_points"] = image_annotation.num_radar_points

        # --------- Ground Truth Annotation ---------
        if self._return_ground_truth:
            self._prepare_annotation(training_sample, sequence_idx, in_sequence_idx)

    def _prepare_annotation(self, training_sample: SampleDataGroup, sequence_idx: int, in_sequence_idx: int):
        '''Prepare the annotation data.

        Args:
            training_sample: Training sample to fill with data. The data will be filled
                in-place into the sample.
            sequence_idx: Index of the sequence.
            in_sequence_idx: Index of the sample in the sequence.
        '''
        nuscenes_sample = self._data.get_sample_in_sequence(sequence_idx, in_sequence_idx)

        annotation = self._converter.prepare_annotation_data_lidar(nuscenes_sample)

        training_sample_annotation = training_sample["gt_boxes"]
        training_sample_annotation["categories"] = annotation.categories
        training_sample_annotation["sizes"] = annotation.sizes
        training_sample_annotation["rotations"] = annotation.rotations
        training_sample_annotation["translations"] = annotation.translations
        training_sample_annotation["num_lidar_points"] = annotation.num_lidar_points
        training_sample_annotation["num_radar_points"] = annotation.num_radar_points
        training_sample_annotation["visibility_level"] = annotation.visibility_levels
        training_sample_annotation["orientations"] = annotation.orientations
        training_sample_annotation["velocities"] = annotation.velocities

    @staticmethod
    def _prepare_needed_sample_data_structure(num_cams: int, return_ground_truth: bool) -> SampleDataGroup:
        '''Prepare the needed sample data structure.

        Args:
            num_cams: Number of cameras.
            return_ground_truth: Whether to return the ground truth data.

        Returns:
            :class:`SampleDataGroup` blueprint describing the sample data structure.
        '''

        ego_pose_geom = SampleDataGroup()
        ego_pose_geom.add_data_field("lidar_ego_pose", types.DALIDataType.FLOAT)
        ego_pose_geom.add_data_field("lidar_ego_pose_inv", types.DALIDataType.FLOAT)

        cam_geom = SampleDataGroup()
        cam_geom.add_data_field("extr_lidar2img", types.DALIDataType.FLOAT)
        cam_geom.add_data_field("intr_lidar2img", types.DALIDataType.FLOAT)
        cam_geom.add_data_field("lidar2img", types.DALIDataType.FLOAT)

        cam = SampleDataGroup()
        cam.add_data_group_field("cam_geometry", cam_geom)
        cam.add_data_field("image", types.DALIDataType.UINT8)
        cam.add_data_field("timestamp", types.DALIDataType.FLOAT64)
        if return_ground_truth:
            gt_boxes_2d = SampleDataGroup()
            gt_boxes_2d.add_data_field("bboxes", types.DALIDataType.FLOAT)
            gt_boxes_2d.add_data_field("centers", types.DALIDataType.FLOAT)
            gt_boxes_2d.add_data_field("depths", types.DALIDataType.FLOAT)
            gt_boxes_2d.add_data_field(
                "categories", types.DALIDataType.INT64, NuscenesStreamPETRDataProvider._mapping_caterogies_2d
            )
            gt_boxes_2d.add_data_field("num_lidar_points", types.DALIDataType.INT32)
            gt_boxes_2d.add_data_field("num_radar_points", types.DALIDataType.INT32)
        cam.add_data_group_field("gt_boxes_2d", gt_boxes_2d)

        res = SampleDataGroup()
        res.add_data_group_field("ego_pose_geom", ego_pose_geom)
        res.add_data_group_field_array("cams", cam, num_cams)
        res.add_data_field("timestamp", types.DALIDataType.FLOAT64)
        res.add_data_field("prev_exists", types.DALIDataType.BOOL)

        if return_ground_truth:
            gt_boxes = SampleDataGroup()
            gt_boxes.add_data_field(
                "categories", types.DALIDataType.INT64, NuscenesStreamPETRDataProvider._mapping_categories_3d
            )
            gt_boxes.add_data_field("sizes", types.DALIDataType.FLOAT)
            gt_boxes.add_data_field("rotations", types.DALIDataType.FLOAT)
            gt_boxes.add_data_field("translations", types.DALIDataType.FLOAT)
            gt_boxes.add_data_field("num_lidar_points", types.DALIDataType.INT32)
            gt_boxes.add_data_field("num_radar_points", types.DALIDataType.INT32)
            gt_boxes.add_data_field(
                "visibility_level",
                types.DALIDataType.INT32,
                NuscenesStreamPETRDataProvider._mapping_visibility,
            )
            gt_boxes.add_data_field("orientations", types.DALIDataType.FLOAT)
            gt_boxes.add_data_field("velocities", types.DALIDataType.FLOAT)

            res.add_data_group_field("gt_boxes", gt_boxes)

        return res
