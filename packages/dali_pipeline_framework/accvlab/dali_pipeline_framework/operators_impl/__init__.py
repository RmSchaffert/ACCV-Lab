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
Operators and utilities for use inside DALI pipelines.

This module contains:
  - Functions to be used inside Python operators
  - Numba operators which can be used directly (including the needed Numba functions and the wrapping as a Numba operator)
'''

from . import numba_operators
from . import python_operator_functions

__all__ = ["numba_operators", "python_operator_functions"]
