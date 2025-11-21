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

# Used to enable type hints using a class type inside the implementation of that class itself.
from __future__ import annotations

from typing import Union, Sequence

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..pipeline.sample_data_group import SampleDataGroup

from .data_groups_with_name_applied_step import DataGroupsWithNameAppliedStep
from .pipeline_step_base import PipelineStepBase


class DataGroupArrayWithNameElementsAppliedStep(DataGroupsWithNameAppliedStep):
    '''Apply a contained processing step independently to each element of all array data group fields with a given name.

    The name is defined at construction. All fields with that name must be array data group fields (see
    :class:`SampleDataGroup`). Each element of each found array is processed independently by the contained step.
    '''

    def __init__(
        self,
        processing_step_to_apply: PipelineStepBase,
        name_of_arrays_to_apply_to: Union[str, int],
        check_minimum_one_name_match=True,
    ):
        '''

        Args:
            processing_step_to_apply: Contained processing step to apply.
            name_of_arrays_to_apply_to: Name of the array data group fields whose elements should be processed.
            check_minimum_one_name_match: If ``True``, require that at least one array is found; otherwise an
                error is raised when checking the input.
        '''

        assert isinstance(
            name_of_arrays_to_apply_to, (str, int)
        ), f"Parameter `name_of_arrays_to_apply_to` has to be of type `str` or `int`. Got `{type(name_of_arrays_to_apply_to)}` instead."

        DataGroupsWithNameAppliedStep.__init__(
            self, processing_step_to_apply, name_of_arrays_to_apply_to, check_minimum_one_name_match
        )

    @override
    def _check_and_get_paths_to_apply_to(self, data: SampleDataGroup) -> Sequence[Sequence[str, int]]:

        array_paths = DataGroupsWithNameAppliedStep._check_and_get_paths_to_apply_to(self, data)
        array_element_paths = []
        for ap in array_paths:
            array_field = data.get_item_in_path(ap)
            if not array_field.is_data_group_field_array():
                raise ValueError(
                    f"DataGroupArraysWithNameElementsAppliedStep: item in path `{ap}` is not a data group field array."
                )
            for i in range(len(array_field)):
                path_to_element = ap + (i,)
                array_element_paths.append(path_to_element)
        return array_element_paths
