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

from typing import Sequence

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase

from ..operators_impl import numba_operators


class CoordinateCropper(PipelineStepBase):
    '''Crop points to a given axis-aligned box.'''

    def __init__(
        self, points_fields_name: str, minimum_point: Sequence[float], maximum_point: Sequence[float]
    ):
        '''

        Args:
            points_fields_name: Name of the data field containing the points to crop. If multiple fields
                with that name are present, each is processed independently.
            minimum_point: Lower corner (min per dimension) of the crop box.
            maximum_point: Upper corner (max per dimension) of the crop box.
        '''

        self._points_fields_name = points_fields_name
        self._minimum_point = minimum_point
        self._maximum_point = maximum_point

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        paths = data.find_all_occurrences(self._points_fields_name)
        for path in paths:
            parent = data.get_parent_of_path(path)
            points = parent[self._points_fields_name]
            dali_dtype = parent.get_type_of_field(self._points_fields_name)
            np_dtype = SampleDataGroup.get_numpy_type_for_dali_type(dali_dtype)
            minimum_point = np.array(self._minimum_point, dtype=np_dtype)
            maximum_point = np.array(self._maximum_point, dtype=np_dtype)
            cropped_points = numba_operators.crop_coordinates(
                points, minimum_point, maximum_point, dali_dtype
            )
            parent[self._points_fields_name] = cropped_points
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        paths = data_empty.find_all_occurrences(self._points_fields_name)
        if len(paths) == 0:
            raise ValueError(
                f"No fields containing points to crop found (searched under name '{self._points_fields_name}')."
            )
        return data_empty
