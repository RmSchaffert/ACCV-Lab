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

from typing import Union, List, Tuple, Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.fn as fn

from .pipeline_step_base import PipelineStepBase
from ..pipeline.sample_data_group import SampleDataGroup


class PaddingToUniform(PipelineStepBase):
    '''Processing step for padding all data fields in the processed data to have the same shape across the batch.

    Padding can be performed either for all data fields, or only for fields with given names.

    Note:
        To pad all fields in a given part (sub-tree) of the input data structure, use the access modifier
        wrapper steps (see :class:`GroupToApplyToSelectedStepBase` and its subclasses).

    '''

    def __init__(
        self,
        field_names: Optional[Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]]] = None,
        fill_value: Union[int, float] = 0.0,
    ):
        '''

        Args:
            field_names: Names of the fields to apply padding to. Can be either a single name or a list of names. All fields with those names are processed.
                If set to ``None``, padding is performed for all data fields. Default is ``None``. Fields can be either data fields or data field arrays.
            fill_value: Value to insert into the padded region. Default is 0.0.
        '''

        if isinstance(field_names, str) or isinstance(field_names, int):
            field_names = [field_names]
        self._field_names = field_names
        self._fill_value = fill_value

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        if self._field_names is None:
            data.ensure_uniform_size_in_batch(self._fill_value)
        else:
            for fnm in self._field_names:
                paths = data.find_all_occurrences(fnm)
                for pth in paths:
                    parent = data.get_parent_of_path(pth)
                    to_pad = parent[fnm]
                    if parent.is_data_group_field(fnm):
                        to_pad.ensure_uniform_size_in_batch(self._fill_value)
                    else:
                        padded = fn.pad(to_pad, fill_value=self._fill_value)
                        parent[fnm] = padded
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        if self._field_names is not None:
            for fn in self._field_names:
                paths = data_empty.find_all_occurrences(fn)
                if len(paths) == 0:
                    raise ValueError(
                        f"No fields to be padded with name '{fn}' were found in the data.\n The format is:\n{data_empty}"
                    )
        return data_empty
