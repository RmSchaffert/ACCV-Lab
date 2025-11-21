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

from typing import Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.types as types

from accvlab.dali_pipeline_framework.pipeline import SampleDataGroup
from accvlab.dali_pipeline_framework.inputs import DataProvider

from .nuscenes_data_converter import NuScenesDataConverter
from .nuscenes_data import NuScenesData


class Nuscenes2DDetectionDataProvider(DataProvider):
    '''Data provider for the 2D object detection task using NuScenes data.

    Can be used with the available input callables/iterables.

    '''

    _num_cams = 6

    def __init__(
        self, base_dir: str, data: NuScenesData, use_multi_cam_samples: bool, decode_images: bool = False
    ):
        '''

        Args:
            base_dir: Base directory of the NuScenes dataset. Used to obtain images from files.
            data: Dataset meta-data.
            use_multi_cam_samples: Whether to combine all cameras into a single sample. Used for demonstration purposes, as the 2D detection task
                does not typically operate on multiple images.
            decode_images: Whether to decode the images when loading them (using openCV). `False` by default. Should be kept as `False` when using
                the DALI pipeline as the decoding is performed more efficiently in the pipeline.
        '''
        self._data = data
        # Note that `category_length_to_keep=2` is hard-coded here.
        # This class is designed to provide the categories based on the first 2 levels (see `mapping_categories` in the
        # method `_prepare_needed_sample_data_structure`).Therefore, in this class, `2` is the only value which makes sense.
        # As `NuScenesDataConverter` is more general and could be used in different use cases, the value is not hard-coded
        # there, and we pass the `2` here instead.
        self._converter = NuScenesDataConverter(
            base_dir, category_length_to_keep=2, decode_images=decode_images
        )
        self._use_multi_cam_samples = use_multi_cam_samples
        # Keep the blueprint for the output data format here, so that it does not need to be generated every time anew
        # when a sample is returned (the sample is constructed by starting with the blueprint and filling in the values).
        self._sample_data_setup = self._prepare_needed_sample_data_structure(
            use_multi_cam_samples, self._num_cams
        )

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        '''Get the data for a given sample ID.

        Args:
            sample_id: ID of the sample to get the data for.

        Returns:
            Data for the given sample ID.
        '''
        sample_data_structure = self._prepare_needed_sample_data_structure(
            self._use_multi_cam_samples, self._num_cams
        )
        if self._use_multi_cam_samples:
            nuscenes_sample = self._data.get_sample_from_flattened_idx(sample_id)
            for i in range(self._num_cams):
                sample_data_structure[i]["image"] = self._converter.prepare_image_data(nuscenes_sample, i)
                # self._converter.prepare_image_annotation(sample_data_structure[i]["annotation"], nuscenes_sample, i)
                annotation_2d = self._converter.prepare_image_annotation(nuscenes_sample, i)
                self._fill_image_annotation(annotation_2d, sample_data_structure[i]["annotation"])
        else:
            nuscenes_sample_id = sample_id // self._num_cams
            index_cam = sample_id % self._num_cams
            nuscenes_sample = self._data.get_sample_from_flattened_idx(nuscenes_sample_id)
            sample_data_structure["image"] = self._converter.prepare_image_data(nuscenes_sample, index_cam)
            annotation_2d = self._converter.prepare_image_annotation(nuscenes_sample, index_cam)
            self._fill_image_annotation(annotation_2d, sample_data_structure["annotation"])
        return sample_data_structure

    @override
    def get_number_of_samples(self) -> int:
        '''Get the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        '''
        data_len = self._data.total_num_samples
        if not self._use_multi_cam_samples:
            data_len *= self._num_cams
        return data_len

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        '''Get the sample data structure (blueprint).'''
        return self._sample_data_setup.get_empty_like_self()

    @staticmethod
    def _fill_image_annotation(
        image_annotation_obj: NuScenesDataConverter.ImageAnnotation,
        annotation_sample_data_group: SampleDataGroup,
    ):
        '''Fill the image annotation for a given image.

        Args:
            image_annotation_obj: Image annotation as returned from the NuScenes data converter.
            annotation_sample_data_group: Data group node into which the annotation data should be filled. Needs to
                contain the corresponding data fields:
                  ["bboxes", "categories", "num_lidar_points", "num_radar_points", "visibility_levels"]
        '''
        annotation_sample_data_group["bboxes"] = image_annotation_obj.bboxes
        annotation_sample_data_group["categories"] = image_annotation_obj.categories
        annotation_sample_data_group["num_lidar_points"] = image_annotation_obj.num_lidar_points
        annotation_sample_data_group["num_radar_points"] = image_annotation_obj.num_radar_points
        annotation_sample_data_group["visibility_levels"] = image_annotation_obj.visibility_levels

    @staticmethod
    def _prepare_needed_sample_data_structure(is_multi_cam: bool, num_cams: Optional[int] = None):
        '''Prepare the needed sample data structure (blueprint).

        Args:
            is_multi_cam: Whether multi-cam samples should be generated (containing all cameras).
            num_cams: Number of available cameras. If `is_multi_cam==True`, this parameter must be set.
                If `is_multi_cam==False`, it is ignored.

        Returns:
            :class:`SampleDataGroup` blueprint describing the sample data structure.
        '''
        assert (not is_multi_cam) or (
            num_cams is not None and num_cams > 0
        ), "If a multi-camera setup is used, number of cameras needs to be set"

        # Mappings for labels from strings to numeric types.
        # Note that multiple vehicle types (bus, truck, trailer) are mapped to the same numerical label. This is allowed and can be used to combine multiple labels
        # in the original dataset with a single label during training, effectively combining the corresponding categories/class types into one.
        mapping_categories = {
            'human.pedestrian': 0,
            'vehicle.bicycle': 1,
            'vehicle.motorcycle': 2,
            'vehicle.car': 3,
            'vehicle.bus': 4,
            'vehicle.truck': 4,
            'vehicle.trailer': 4,
            'vehicle.construction': 5,
            None: 6,
        }
        mapping_visibility = {'v0-40': 0, 'v40-60': 1, 'v60-80': 3, 'v80-100': 4}

        cam = SampleDataGroup()

        cam.add_data_field("image", types.DALIDataType.UINT8)

        annotation = SampleDataGroup()
        annotation.add_data_field("bboxes", types.DALIDataType.FLOAT)
        # "categories" and "visibility_levels" use the mappings described above to auto-convert strings to numeric values
        # on assignment.
        annotation.add_data_field("categories", types.DALIDataType.INT32, mapping_categories)
        annotation.add_data_field("visibility_levels", types.DALIDataType.INT32, mapping_visibility)
        annotation.add_data_field("num_lidar_points", types.DALIDataType.INT32)
        annotation.add_data_field("num_radar_points", types.DALIDataType.INT32)
        cam.add_data_group_field("annotation", annotation)

        # If multi-cam samples are used, the data format is an array, where each entry represents one camera.
        # Else, it is the data corresponding to one camera.
        if is_multi_cam:
            res = SampleDataGroup.create_data_group_field_array(cam, num_cams)
        else:
            res = cam

        return res
