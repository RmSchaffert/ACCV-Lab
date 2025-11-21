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

from typing import Union, Sequence

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.fn as fn

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase


class AxesLayoutSetter(PipelineStepBase):
    '''Set the DALI axes layout string (e.g., "HWC", "CHW") for selected fields.'''

    def __init__(self, names_fields_to_set: Union[str, int, Sequence[Union[str, int]]], layout_to_set: str):
        '''

        Args:
            names_fields_to_set: Name or list of names of fields for which the layout should be set.
                All matching fields are processed.
            layout_to_set: DALI axes layout string (e.g., "HWC", "CHW")
        '''

        if isinstance(names_fields_to_set, str) or isinstance(names_fields_to_set, int):
            names_fields_to_set = [names_fields_to_set]
        self._names_fields_to_set = names_fields_to_set

        self._layout_to_set = layout_to_set

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Note that only the actual image normalization becomes part of the DALI graph, the search for images as well as resolvong the paths obtained in order to access the image,
        # and the change of the data type, happen at graph construction time and do not influence the run time.

        for field_name in self._names_fields_to_set:
            # Find all fields
            field_paths = data.find_all_occurrences(field_name)

            for fp in field_paths:
                # Get the field
                field = data.get_item_in_path(fp)
                # Adjust layout (use reshape to set layout metadata without changing data)
                field = fn.reshape(field, layout=self._layout_to_set)
                # Set the updated field
                data.set_item_in_path(fp, field)

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        for field_name in self._names_fields_to_set:

            field_paths = data_empty.find_all_occurrences(field_name)

            # Make sure at least 1 field is available
            if len(field_paths) == 0:
                raise KeyError(f"No occurrences of field '{field_name}' found.")

        return data_empty
