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

import pytest
import torch
from accvlab.example_package import external_vector_add_cuda, external_vector_scale_cuda


def test_external_vector_add():
    """Test external vector addition"""
    # Create test tensors
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', dtype=torch.float32)
    b = torch.tensor([5.0, 6.0, 7.0, 8.0], device='cuda', dtype=torch.float32)

    # Test external implementation
    result = external_vector_add_cuda(a, b)

    # Verify result
    expected = torch.tensor([6.0, 8.0, 10.0, 12.0], device='cuda', dtype=torch.float32)
    assert torch.allclose(result, expected)
    assert result.device == a.device
    assert result.dtype == torch.float32


def test_external_vector_scale():
    """Test external vector scaling"""
    # Create test tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', dtype=torch.float32)
    scale = 2.5

    # Test external implementation
    result = external_vector_scale_cuda(input_tensor, scale)

    # Verify result
    expected = torch.tensor([2.5, 5.0, 7.5, 10.0], device='cuda', dtype=torch.float32)
    assert torch.allclose(result, expected)
    assert result.device == input_tensor.device
    assert result.dtype == torch.float32


if __name__ == "__main__":
    pytest.main()
