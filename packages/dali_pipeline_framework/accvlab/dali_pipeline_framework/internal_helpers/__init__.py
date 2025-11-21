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

'''
This module contains helper functions and classes which are used internally in the package
or can be useful when debugging or implementing custom functionality.

For debugging, :func:`print_tensor_op` and :func:`print_tensor_size_op` can be used to print tensors
from within the DALI pipeline.

The other functionality is used internally, and may be useful when implementing custom processing steps.

'''

from .check_type import check_type
from .debug_helpers import print_tensor_op, print_tensor_size_op
from .helper_functions import get_mapped, get_as_data_node
from . import mini_parser

__all__ = [
    "check_type",
    "print_tensor_op",
    "print_tensor_size_op",
    "get_mapped",
    "get_as_data_node",
    "mini_parser",
]
