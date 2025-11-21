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

from .data_group_in_path_applied_step import DataGroupInPathAppliedStep
from .pipeline_step_base import PipelineStepBase


class DataGroupArrayInPathElementsAppliedStep(DataGroupInPathAppliedStep):
    '''Apply a contained processing step independently to each element of an array data group field at a path.

    The path of the array data group field is defined at construction. Each element of that array is processed
    independently by the contained step.
    '''

    def __init__(
        self,
        processing_step_to_apply: PipelineStepBase,
        path_to_array_to_apply_to: Union[str, int, Sequence[str, int]],
    ):
        '''

        Args:
            processing_step_to_apply: Contained processing step to apply.
            path_to_array_to_apply_to: Path to the array data group field whose children should be processed.

        '''
        DataGroupInPathAppliedStep.__init__(self, processing_step_to_apply, path_to_array_to_apply_to)

    @override
    def _check_and_get_paths_to_apply_to(self, data: SampleDataGroup) -> Sequence[Sequence[str, int]]:

        array_paths = DataGroupInPathAppliedStep._check_and_get_paths_to_apply_to(self, data)
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
