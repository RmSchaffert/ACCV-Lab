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

import nvidia.dali.types as types

from .pipeline_step_base import PipelineStepBase
from ..pipeline.sample_data_group import SampleDataGroup

from ..operators_impl import numba_operators


class PointsInRangeCheck(PipelineStepBase):
    '''Check whether points lie within a given axis-aligned box and add a boolean mask.

    See also:
        - :class:`AnnotationElementConditionEval` can be used to combine the results of this step with other
          conditions.
        - :class:`ConditionalElementRemoval` can be used to remove elements from the data based on this
          condition or a combination of this condition with other conditions.

    '''

    def __init__(
        self,
        points_fields_name: str,
        is_inside_field_name: str,
        minimum_point: Sequence[float],
        maximum_point: Sequence[float],
    ):
        '''

        Args:
            points_fields_name: Name of the data field containing the points to check. If multiple fields
                with that name are present, each is processed independently.
            is_inside_field_name: Name of the sibling data field to store the boolean mask in. Must not
                already exist.
            minimum_point: Lower corner (min per dimension) of the region.
            maximum_point: Upper corner (max per dimension) of the region.
        '''

        self._points_fields_name = points_fields_name
        self._is_inside_field_name = is_inside_field_name
        self._minimum_point = minimum_point
        self._maximum_point = maximum_point

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        paths = data.find_all_occurrences(self._points_fields_name)
        for path in paths:
            parent = data.get_parent_of_path(path)
            points = parent[self._points_fields_name]
            mask = numba_operators.check_points_in_box(points, self._minimum_point, self._maximum_point)
            parent.add_data_field(self._is_inside_field_name, types.DALIDataType.BOOL)
            parent[self._is_inside_field_name] = mask
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        paths = data_empty.find_all_occurrences(self._points_fields_name)
        if len(paths) == 0:
            raise ValueError(
                f"No fields containing points to check found (searched under name '{self._points_fields_name}')."
            )
        for path in paths:
            parent = data_empty.get_parent_of_path(path)
            if parent.has_child(self._is_inside_field_name):
                raise ValueError(
                    f"Cannot add is_inside flag as field with name '{self._is_inside_field_name}' as a sibling with that name is is already present for points at path `{path}`."
                )
            parent.add_data_field(self._is_inside_field_name, types.DALIDataType.BOOL)
        return data_empty
