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

from abc import abstractmethod

from ..pipeline.sample_data_group import SampleDataGroup
from .pipeline_step_base import PipelineStepBase


class GroupToApplyToSelectedStepBase(PipelineStepBase):
    '''Base class for wrappers that apply a contained processing step to selected parts (sub-trees) of the input.

    The wrapper forwards only the selected parts (sub-tree(s)) to the contained step, which then operates as
    if the sub-tree were the full input. If multiple sub-trees are selected (e.g. each sub-tree corresponding
    to data of one step in time out of a sequence), the contained step is called multiple times, executing
    independently for each sub-tree. If joint processing is required, design the contained step to consume
    the full tree (or a larger sub-tree) instead of using a wrapper.

    Args:
        processing_step_to_apply: Processing step to apply to the selected sub-trees.

    Important:

        Ensure that the constructor of this class is called by any derived class.

    .. automethod:: GroupToApplyToSelectedStepBase._check_and_get_paths_to_apply_to

    '''

    def __init__(self, processing_step_to_apply):
        # Note that the argument documentation is directly written in the class docstring. As the
        # docstring has an automethod, this is more readable. As the docstring of __init__ is concatenated
        # after the class docstring, the automethod would need to be added here otherwise.

        self._processing_step_to_apply = processing_step_to_apply

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        paths = self._check_and_get_paths_to_apply_to(data)
        for path in paths:
            data_to_apply_to = data.get_item_in_path(path)
            applied = self._processing_step_to_apply(data_to_apply_to)
            data.change_type_of_data_and_remove_data(path, applied)
            data.set_item_in_path(path, applied)
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        paths = self._check_and_get_paths_to_apply_to(data_empty)
        for path in paths:
            data_empty_to_apply_to = data_empty.get_item_in_path(path)
            applied = self._processing_step_to_apply.check_input_data_format_and_set_output_data_format(
                data_empty_to_apply_to
            )
            data_empty.change_type_of_data_and_remove_data(path, applied)
            data_empty.set_item_in_path(path, applied)
        return data_empty

    @abstractmethod
    def _check_and_get_paths_to_apply_to(self, data: SampleDataGroup) -> Sequence[Sequence[str, int]]:
        '''Check input and return paths to all sub-trees to process.

        Requirements on the input include that at least one sub-path is found and that paths match the
        expected type (e.g., array data group fields when iterating over elements).
        See :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` for what constitutes an array and how to check whether a field is an
        array.

        If the requirements are not satisfied, an error shall be raised.

        Note:

            Override this method in each (non-abstract) derived class to define the actual selection
            of sub-trees to process. Note that this is the only method which needs to be overridden,
            and is used by the other methods of this class, which perform the actual processing.

        '''
        pass
