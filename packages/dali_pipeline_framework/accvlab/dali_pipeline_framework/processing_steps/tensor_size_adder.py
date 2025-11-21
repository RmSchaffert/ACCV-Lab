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
import nvidia.dali.fn as fn

from .pipeline_step_base import PipelineStepBase
from ..pipeline.sample_data_group import SampleDataGroup


class TensorSizeAdder(PipelineStepBase):
    '''Add tensor size fields for tensors by name using DALI's dynamic execution model.

    This step finds all tensors with the specified name and adds a sibling field containing the
    tensor's size (height and width, i.e., dimensions -3 and -2) using DALI's `.shape` property.
    The size is stored on CPU with the configured numeric type.

    Behavior:
      - Finds all tensors by name and extracts their size using `.shape`.
      - Adds a sibling field with the name ``tensor_name + size_postfix`` containing the
        tensor size as [height, width] on CPU with the configured numeric type for each tensor.
      - Tensor search and field creation happen at graph construction time; only obtaining the size
        via `.shape` is part of the DALI graph.
    '''

    def __init__(
        self,
        tensor_name: str,
        size_postfix: str,
        store_size_as_type: types.DALIDataType = types.DALIDataType.INT32,
    ):
        '''

        Args:
            tensor_name: Name of the tensor data field(s) to extract size from
            size_postfix: Postfix to be added to ``tensor_name`` and used as the name when storing the
                tensor size. Stored as a sibling to the corresponding tensor.
            store_size_as_type: Element data type for the added field containing the tensor size.
        '''

        self._tensor_name = tensor_name
        self._size_postfix = size_postfix
        self._store_size_as_type = store_size_as_type

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Find all tensors with the specified name
        tensor_paths = data.find_all_occurrences(self._tensor_name)

        # Process each tensor
        for tp in tensor_paths:
            # Get the tensor
            tensor = data.get_item_in_path(tp)

            # Extract the size using DALI's .shape() method
            # For images/tensors, dimensions -3 and -2 correspond to height and width
            tensor_shape = tensor.shape()
            # Get height (dimension -3) and width (dimension -2)
            # Use fn.stack to create a proper tensor with [height, width]
            # Cast to the desired output type
            tensor_hw = fn.cast(fn.stack(tensor_shape[-3], tensor_shape[-2]), dtype=self._store_size_as_type)

            # Get the parent and add the size field
            parent = data.get_parent_of_path(tp)
            size_field_name = self._tensor_name + self._size_postfix

            # Add the sibling field for the tensor size
            parent.add_data_field(size_field_name, self._store_size_as_type)
            parent[size_field_name] = tensor_hw

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        # Find all tensors with the specified name
        tensor_paths = data_empty.find_all_occurrences(self._tensor_name)

        if len(tensor_paths) == 0:
            raise KeyError(
                f"No occurrences of tensors found. Fields containing tensors are expected to have the name "
                f"'{self._tensor_name}', as specified in the constructor."
            )

        # For each tensor found, add the size field
        for tp in tensor_paths:
            tensor_parent = data_empty.get_parent_of_path(tp)
            size_field_name = self._tensor_name + self._size_postfix

            # Add the size field to the blueprint
            # This will raise an exception if the field already exists
            tensor_parent.add_data_field(size_field_name, self._store_size_as_type)

        return data_empty
