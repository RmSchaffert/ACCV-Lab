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

"""
ACCV-Lab Examples Functions
Wrapper functions for C++ and CUDA extensions.
"""

from __future__ import annotations

# Import extensions (assuming they are installed correctly)
# Note that torch needs to be imported for the extensions import to work
import torch

from torch import Tensor

import accvlab.example_package._cpp as _cpp
import accvlab.example_package._cuda as _cuda


# C++ wrapper functions
def cpp_vector_sum(vector: Tensor) -> float:
    """
    Sum all elements of a 1D tensor using the C++ extension.

    :cpu:

    Args:
        vector: 1D tensor whose elements will be summed. Must be a contiguous
            :class:`torch.Tensor` on a device supported by the extension.

    Returns:
        The sum of all elements in ``vector``.
    """
    return _cpp.vector_sum(vector)


def cpp_matrix_transpose(matrix: Tensor) -> Tensor:
    """
    Transpose a 2D tensor using the C++ extension.

    :cpu:

    Args:
        matrix: 2D tensor to transpose. Must be a contiguous :class:`torch.Tensor`.

    Returns:
        A new tensor containing the transpose of ``matrix``.
    """
    return _cpp.matrix_transpose(matrix)


def cpp_print_build_info() -> None:
    """
    Print build information for the C++ extension to standard output.

    This is mainly intended for debugging to verify that the extension
    was built with the expected compiler flags and configuration.

    Returns:
        None
    """
    _cpp.print_build_info()


def cpp_vector_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two 1D tensors element‑wise using the C++ extension.

    :cpu:

    Args:
        a: First input 1D tensor.
        b: Second input 1D tensor. Must be broadcast‑compatible with ``a``.

    Returns:
        Tensor containing the element‑wise product of ``a`` and ``b``.
    """
    return _cpp.vector_multiply(a, b)


# CUDA wrapper functions
def cuda_vector_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two 1D tensors element‑wise using the CUDA extension.

    :gpu:

    Args:
        a: First input 1D tensor on a CUDA device.
        b: Second input 1D tensor on a CUDA device. Must be broadcast‑compatible
            with ``a``.

    Returns:
        Tensor on CUDA containing the element‑wise product of ``a`` and ``b``.
    """
    return _cuda.vector_multiply(a, b)


def cuda_reduce_sum(input: Tensor) -> Tensor:
    """
    Compute the sum of all elements in a tensor using a CUDA reduction kernel.

    :gpu:

    Args:
        input: Input tensor on a CUDA device. Can be 1D or higher‑dimensional;
            all elements are reduced to a single value.

    Returns:
        0‑dimensional tensor (scalar) on CUDA containing the sum of all
        elements in ``input``.
    """
    return _cuda.reduce_sum(input)
