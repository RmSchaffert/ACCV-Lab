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

from typing import Union, List, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase


class UnneededFieldRemover(PipelineStepBase):
    '''Processing step for removing unneeded fields from the data.

    This step does not add any processing steps to the DALI graph, i.e. it is fully performed on DALI graph
    construction time and does not have any overhead at runtime. This means that this step can be used inside
    the pipeline multiple times to ensure a clean data structure without any performance penalty (apart from
    the overhead at graph construction time).

    Note:
        For pipelines which use data which is not needed in the final output (e.g. intermediate results,
        image size on the CPU, etc.), it is advisable to perform this step at least once, directly before
        outputting the data, in order to avoid unneeded copies & clutter in the final output.
    '''

    def __init__(self, unneeded_field_names: Union[Tuple[Union[str, int], ...], List[Union[str, int]]]):
        '''
        Args:
            unneeded_field_names: Names of the fields to be removed. All fields with those names are removed.

        '''

        assert isinstance(unneeded_field_names, tuple) or isinstance(
            unneeded_field_names, list
        ), f"`unneeded_field_names` has to be a tuple or a list; got: {type(unneeded_field_names)}"

        self._unneeded_field_names = unneeded_field_names

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # The actual processing (i.e. `__call__()`) is exactly the same as what `_check_and_adjust_data_format_input_to_output()` needs to do. Therefore, implement it
        # in this method and call it from both of the mentioned methods.
        for ufn in self._unneeded_field_names:
            data.remove_all_occurrences(ufn)
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        return self._process(data_empty)
