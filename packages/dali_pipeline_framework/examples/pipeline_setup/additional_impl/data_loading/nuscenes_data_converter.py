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

import numpy as np
from pyquaternion import Quaternion

from .nuscenes_data import NuScenesDataSample


class NuScenesDataConverter:
    '''Converter for NuScenes data.

    The converter contains helper functions to convert the NuScenes data to a format
    which can be used in the DALI pipelines. It is not specific to a single task, but instead,
    can be used by the individual data providers to help with data preparation and to avoid
    re-implementing the same functionality for different tasks (i.e. in the different data
    providers).
    '''

    class ImageAnnotation:
        def __init__(
            self,
            bboxes: np.ndarray,
            categories: List[str],
            visibility_levels: List[str],
            num_lidar_points: np.ndarray,
            num_radar_points: np.ndarray,
        ):
            '''

            Args:
                bboxes: Bounding boxes. Shape: (num_bboxes, 4).
                    Each row contains the upper left and the lower right corner of the bounding box.
                categories: Categories of the objects.
                visibility_levels: Visibility levels of the objects.
                    The format is as used in the NuScenes dataset.
                num_lidar_points: Number of lidar points. Shape: (num_bboxes,).
                num_radar_points: Number of radar points. Shape: (num_bboxes,).
            '''

            self.bboxes = bboxes
            self.categories = categories
            self.visibility_levels = visibility_levels
            self.num_lidar_points = num_lidar_points
            self.num_radar_points = num_radar_points

    class ProjectedBboxesAnnotation:
        def __init__(
            self,
            bboxes: np.ndarray,
            centers: np.ndarray,
            categories: List[str],
            num_lidar_points: np.ndarray,
            num_radar_points: np.ndarray,
            depths: np.ndarray,
        ):
            '''

            Args:
                bboxes: Bounding boxes. Shape: (num_bboxes, 4).
                    Each row contains the upper left and the lower right corner of the bounding box.
                centers: Centers. Shape: (num_bboxes, 2).
                    Each row contains the center of the bounding box.
                categories: Categories of the objects.
                num_lidar_points: Number of lidar points. Shape: (num_bboxes,).
                num_radar_points: Number of radar points. Shape: (num_bboxes,).
                depths: Depths of the bounding box centers. Shape: (num_bboxes,).
            '''
            self.bboxes = bboxes
            self.centers = centers
            self.categories = categories
            self.num_lidar_points = num_lidar_points
            self.num_radar_points = num_radar_points
            self.depths = depths

    class Annotation:
        def __init__(
            self,
            categories: List[str],
            sizes: np.ndarray,
            rotations: np.ndarray,
            translations: np.ndarray,
            visibility_levels: List[str],
            num_lidar_points: np.ndarray,
            num_radar_points: np.ndarray,
            orientations: np.ndarray,
            velocities: np.ndarray,
        ):
            '''

            Args:
                categories: Categories of the objects.
                sizes: Sizes of the objects. Shape: (num_bboxes, 3).
                    The format is as used in the NuScenes dataset.
                rotations: Rotations of the objects. Shape: (num_bboxes, 4).
                    This is a quaternion. The format is as used in the NuScenes dataset.
                translations: Translations of the objects. Shape: (num_bboxes, 3).
                    The format is as used in the NuScenes dataset.
                visibility_levels: Visibility levels of the objects.
                    The format is as used in the NuScenes dataset.
                num_lidar_points: Number of lidar points. Shape: (num_bboxes,).
                num_radar_points: Number of radar points. Shape: (num_bboxes,).
                orientations: Orientations of the objects. Shape: (num_bboxes,).
                    This is the yaw angle of the rotation.
                velocities: Velocities of the objects. Shape: (num_bboxes, 3).
            '''
            self.categories = categories
            self.sizes = sizes
            self.rotations = rotations
            self.translations = translations
            self.num_lidar_points = num_lidar_points
            self.num_radar_points = num_radar_points
            self.visibility_levels = visibility_levels
            self.orientations = orientations
            self.velocities = velocities

    def __init__(self, base_dir, category_length_to_keep, decode_images=False):
        '''

        Args:
            base_dir: Base directory of the NuScenes dataset.
            category_length_to_keep: Length of the category string to keep.
                This is used to limit the sub-category level to keep.
            decode_images: Whether to decode the images. Note that typically,
                the images should not be docoded, as this is done efficiently
                inside the DALI pipeline.
        '''

        self._base_dir = base_dir
        self._category_length = category_length_to_keep
        self._decode_images = decode_images
        if decode_images:
            try:
                import cv2
            except ImportError:
                raise ImportError(
                    "cv2 is not installed. It is needed if `decode_images` is set to `True` in the constructor."
                )
            self._cv2 = cv2

    def prepare_image_data(self, nuscenes_sample: NuScenesDataSample, cam_id: int) -> np.ndarray:
        '''Prepare the image data for a given camera and sample.

        Args:
            nuscenes_sample: NuScenes sample.
            cam_id: Camera index.

        Returns:
            The resulting image data.
        '''

        cam_sample = nuscenes_sample.camera_samples[cam_id]
        # cam_sample = self._data.samples[sample_id].camera_samples[cam_id]
        filename = self._base_dir + '/' + cam_sample['filename']
        if not self._decode_images:
            with open(filename, "rb") as f:
                image_data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            image_data = self._cv2.imread(filename, self._cv2.IMREAD_COLOR)
        return image_data

    def prepare_image_annotation(self, nuscenes_sample: NuScenesDataSample, cam_id: int) -> ImageAnnotation:
        '''Prepare the image annotation data for a given camera and sample.

        Note:
            The image annotations are as provided by the NuScenes projection script.

        Args:
            nuscenes_sample: NuScenes sample.
            cam_id: Camera index.

        Returns:
            The resulting image annotation data.
        '''

        annotations = nuscenes_sample.image_annotations[cam_id]

        # For the bboxes, it is important to ensure the correct shape, even if there are no objects.
        # If this is not done, the resulting array is 2D if data is present, but is dimensionless if no data is present.
        # This would lead to problems when outputting the batch of data, as here, the individual outputs need to
        # have matching dimensions
        bboxes = np.array([ann['bbox_corners'] for ann in annotations], dtype=float).reshape([-1, 4])
        categories = [
            self._shorten_category(ann['category_name'], self._category_length) for ann in annotations
        ]
        visibility_levels = [ann['visibility_level']['level'] for ann in annotations]
        num_lidar_points = np.array([ann['num_lidar_pts'] for ann in annotations], dtype=np.int32)
        num_radar_points = np.array([ann['num_radar_pts'] for ann in annotations], dtype=np.int32)

        res = self.ImageAnnotation(bboxes, categories, visibility_levels, num_lidar_points, num_radar_points)
        return res

    def prepare_projected_bboxes_annotations(
        self, nuscenes_sample: NuScenesDataSample, cam_id: int
    ) -> ProjectedBboxesAnnotation:
        '''Prepare the projected bboxes annotation data for a given camera and sample.

        The projections are generated by :class:`BboxProjector` class.

        Args:
            nuscenes_sample: NuScenes sample.
            cam_id: Camera index.

        Returns:
            The resulting projected bboxes annotation data.
        '''

        if nuscenes_sample.projected_bboxes_annotations is None:
            raise ValueError(
                "No projected bboxes annotations are present in the `NuScenesDataSample` object. "
                "Please configure the `NuScenesReader` to add projected bboxes annotations."
            )

        annotations = nuscenes_sample.projected_bboxes_annotations[cam_id]

        categories_adjusted = [
            self._shorten_category(cat, self._category_length) for cat in annotations['categories']
        ]
        res = self.ProjectedBboxesAnnotation(
            annotations['bboxes'],
            annotations['centers'],
            categories_adjusted,
            annotations['num_lidar_pts'],
            annotations['num_radar_pts'],
            annotations['depths'],
        )
        return res

    def prepare_annotation_data_global(self, nuscenes_sample: NuScenesDataSample) -> Annotation:
        '''Prepare the annotation data for a given sample.

        Args:
            nuscenes_sample: NuScenes sample.

        Returns:
            The resulting annotation data.
        '''

        annotations = nuscenes_sample.annotations
        visibility_annotations = nuscenes_sample.visibilities
        velocities = nuscenes_sample.annotation_velocities

        # Note that the `.reshape` calls below are needed in case that there are no objects.
        # If this is not done, the resulting array is 2D if data is present, but is dimensionless if no data is present.
        # This would lead to problems when outputting the batch of data, as here, the individual outputs need to
        # have matching dimensions
        categories = [
            self._shorten_category(ann['category_name'], self._category_length) for ann in annotations
        ]
        sizes = np.array([ann["size"] for ann in annotations]).reshape([-1, 3])
        rotations = np.array([ann["rotation"] for ann in annotations]).reshape([-1, 4])
        translations = np.array([ann["translation"] for ann in annotations]).reshape([-1, 3])
        num_lidar_points = np.array([ann["num_lidar_pts"] for ann in annotations])
        num_radar_points = np.array([ann["num_radar_pts"] for ann in annotations])
        visibilities = [vis_ann["level"] for vis_ann in visibility_annotations]
        orientations = np.array([self._get_orientation(rot) for rot in rotations])
        res = self.Annotation(
            categories,
            sizes,
            rotations,
            translations,
            visibilities,
            num_lidar_points,
            num_radar_points,
            orientations,
            velocities,
        )
        return res

    def prepare_annotation_data_lidar(self, nuscenes_sample: NuScenesDataSample) -> Annotation:
        '''Prepare the annotation data for a given sample.

        Args:
            nuscenes_sample: NuScenes sample.

        Returns:
            The resulting annotation data.
        '''

        annotations = nuscenes_sample.annotations
        objects_lidar = nuscenes_sample.objects_in_lidar_coords
        visibility_annotations = nuscenes_sample.visibilities

        categories = [self._shorten_category(obj.name, self._category_length) for obj in objects_lidar]
        sizes = np.array([obj.wlh for obj in objects_lidar]).reshape([-1, 3])
        rotations = np.array([obj.orientation.elements for obj in objects_lidar]).reshape([-1, 4])
        translations = np.array([obj.center for obj in objects_lidar]).reshape([-1, 3])
        num_lidar_points = np.array([ann["num_lidar_pts"] for ann in annotations])
        num_radar_points = np.array([ann["num_radar_pts"] for ann in annotations])
        visibilities = [vis_ann["level"] for vis_ann in visibility_annotations]
        orientations = np.array([obj.orientation.yaw_pitch_roll[0] for obj in objects_lidar])
        velocities = np.array([obj.velocity for obj in objects_lidar])

        res = self.Annotation(
            categories,
            sizes,
            rotations,
            translations,
            visibilities,
            num_lidar_points,
            num_radar_points,
            orientations,
            velocities,
        )
        return res

    @staticmethod
    def _get_orientation(rotation: np.ndarray) -> float:
        '''Get the orientation for a given rotation quaternion.

        The orientation is the yaw angle of the rotation.

        Args:
            rotation: Rotation.

        Returns:
            The resulting orientation.
        '''

        orientation = Quaternion(rotation).yaw_pitch_roll[0]
        return orientation

    @staticmethod
    def _shorten_category(category: str, num_levels: int) -> str:
        '''Shorten the category string to the given number of levels.

        Args:
            category: Category string.
            num_levels: Number of levels to keep.

        Returns:
            The shortened category string.
        '''

        points = [i for i, c in enumerate(category) if c == '.']
        if len(points) >= num_levels:
            res = category[: points[num_levels - 1]]
        else:
            res = category
        return res
