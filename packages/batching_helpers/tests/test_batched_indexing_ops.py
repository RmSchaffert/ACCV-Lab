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
import numpy as np

from accvlab.batching_helpers import (
    RaggedBatch,
    batched_indexing_access,
    batched_inverse_indexing_access,
    batched_indexing_write,
)

# -------------------------------------------------------------------------------------------------
# Manual test data generation
# -------------------------------------------------------------------------------------------------


def _create_manual_test_data(dtype: torch.dtype = torch.float32):
    """Creates input data and indices for manual testing of batched indexing operations.

    Returns:
        tuple: (input_data, index_batch, filler_value, expected_output, expected_grad)
            - input_data: Tensor of shape (3, 5, 2) with distinct values
            - index_batch: RaggedBatch with manually specified indices
            - filler_value: Float value used for filling
            - expected_output: Expected output tensor for forward pass
            - expected_grad: Expected gradient tensor for backward pass
    """
    batch_shape = (2, 3)
    total_batch_size = np.prod(batch_shape)
    num_inputs = 5
    num_outputs = 3
    scalar_type = dtype
    filler_value = -4.1

    # Create input data with distinct values for easy verification
    input_data = torch.arange(total_batch_size * num_inputs * 2, dtype=scalar_type, device="cuda:0")
    # Note that the input data has 2 channels
    input_data = input_data.reshape(*batch_shape, num_inputs, 2)

    # Create non-uniform indices and sample sizes
    nums_elements = torch.tensor([[2, 3, 1], [3, 2, 1]], dtype=torch.int64, device="cuda:0")
    # Initialize the indices with a large value to ensure that if invalid indices are used,
    # an error will be produced.
    indices_data = torch.full((*batch_shape, num_outputs), 100, dtype=torch.int64, device="cuda:0")

    # Sample 0, 0: 2 indices
    indices_data[0, 0, 0:2] = torch.tensor([1, 3], dtype=torch.int64, device="cuda:0")
    # Sample 0, 1: 3 indices
    indices_data[0, 1, 0:3] = torch.tensor([0, 2, 4], dtype=torch.int64, device="cuda:0")
    # Sample 0, 2: 1 index
    indices_data[0, 2, 0:1] = torch.tensor([2], dtype=torch.int64, device="cuda:0")

    # Sample 1, 0: 3 indices
    indices_data[1, 0, 0:3] = torch.tensor([1, 3, 2], dtype=torch.int64, device="cuda:0")
    # Sample 1, 1: 2 indices
    indices_data[1, 1, 0:2] = torch.tensor([2, 1], dtype=torch.int64, device="cuda:0")
    # Sample 1, 2: 1 index
    indices_data[1, 2, 0:1] = torch.tensor([1], dtype=torch.int64, device="cuda:0")
    index_batch = RaggedBatch(indices_data, sample_sizes=nums_elements)

    # Create expected output for forward pass
    expected_output = torch.full((2, 3, 3, 2), filler_value, dtype=scalar_type, device="cuda:0")
    expected_output[0, 0, 0] = input_data[0, 0, 1]
    expected_output[0, 0, 1] = input_data[0, 0, 3]
    expected_output[0, 1, 0] = input_data[0, 1, 0]
    expected_output[0, 1, 1] = input_data[0, 1, 2]
    expected_output[0, 1, 2] = input_data[0, 1, 4]
    expected_output[0, 2, 0] = input_data[0, 2, 2]

    expected_output[1, 0, 0] = input_data[1, 0, 1]
    expected_output[1, 0, 1] = input_data[1, 0, 3]
    expected_output[1, 0, 2] = input_data[1, 0, 2]
    expected_output[1, 1, 0] = input_data[1, 1, 2]
    expected_output[1, 1, 1] = input_data[1, 1, 1]
    expected_output[1, 2, 0] = input_data[1, 2, 1]

    # Create expected gradient for backward pass only for floating-point types
    if input_data.dtype.is_floating_point:
        expected_grad = torch.zeros_like(input_data)
        expected_grad[0, 0, 1] = torch.cos(input_data[0, 0, 1])
        expected_grad[0, 0, 3] = torch.cos(input_data[0, 0, 3])
        expected_grad[0, 1, 0] = torch.cos(input_data[0, 1, 0])
        expected_grad[0, 1, 2] = torch.cos(input_data[0, 1, 2])
        expected_grad[0, 1, 4] = torch.cos(input_data[0, 1, 4])
        expected_grad[0, 2, 2] = torch.cos(input_data[0, 2, 2])

        expected_grad[1, 0, 1] = torch.cos(input_data[1, 0, 1])
        expected_grad[1, 0, 3] = torch.cos(input_data[1, 0, 3])
        expected_grad[1, 0, 2] = torch.cos(input_data[1, 0, 2])
        expected_grad[1, 1, 2] = torch.cos(input_data[1, 1, 2])
        expected_grad[1, 1, 1] = torch.cos(input_data[1, 1, 1])
        expected_grad[1, 2, 1] = torch.cos(input_data[1, 2, 1])
    else:
        expected_grad = None

    return input_data, index_batch, filler_value, expected_output, expected_grad


def _create_manual_inverse_test_data(filler_value, dtype: torch.dtype = torch.float32):
    # Create input data
    input_data = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Create indices with variable lengths per sample
    indices = torch.tensor(
        [
            [
                [0, 1, 2, -1],
                [1, 2, -1, -1],
                [2, 0, 1, 4],
            ],
            [
                [3, 0, -1, -1],
                [4, 0, -1, -1],
                [1, -1, -1, -1],
            ],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    # Set the invalid entries to a large value. This value should not be used for any purpose,
    # but setting it to a large value ensures that errors will be prodiced if it is used as an
    # index.
    indices[indices == -1] = 100

    # Create sample sizes (number of valid indices per sample)
    sample_sizes = torch.tensor([[3, 2, 4], [2, 2, 1]], dtype=torch.int64, device="cuda:0")
    indices_batch = RaggedBatch(indices, sample_sizes=sample_sizes)

    # Expected output for forward pass
    fvl = filler_value
    expected_output = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, fvl, fvl],
                [fvl, 5.0, 6.0, fvl, fvl],
                [10.0, 11.0, 9.0, fvl, 12.0],
            ],
            [
                [14.0, fvl, fvl, 13.0, fvl],
                [18.0, fvl, fvl, fvl, 17.0],
                [fvl, 21.0, fvl, fvl, fvl],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Expected gradients for backward pass
    if dtype.is_floating_point:
        nan = torch.nan
        expected_grad = torch.cos(
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 3.0, nan],
                        [5.0, 6.0, nan, nan],
                        [9.0, 10.0, 11.0, 12.0],
                    ],
                    [
                        [13.0, 14.0, nan, nan],
                        [17.0, 18.0, nan, nan],
                        [21.0, nan, nan, nan],
                    ],
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_grad = torch.nan_to_num(expected_grad, nan=0.0)
    else:
        expected_grad = None

    return input_data, indices_batch, expected_output, expected_grad


def _create_manual_write_test_data(dtype: torch.dtype = torch.float32):
    # Create input data
    to_write_data = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Create data to write into
    to_write_into_data = torch.tensor(
        [
            [
                [-1.0, -2.0, -3.0, -4.0, -5.0],
                [-6.0, -7.0, -8.0, -9.0, -10.0],
                [-11.0, -12.0, -13.0, -14.0, -15.0],
            ],
            [
                [-16.0, -17.0, -18.0, -19.0, -20.0],
                [-21.0, -22.0, -23.0, -24.0, -25.0],
                [-26.0, -27.0, -28.0, -29.0, -30.0],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Create indices with variable lengths per sample
    indices = torch.tensor(
        [
            [
                [0, 1, 2, -1],
                [1, 2, -1, -1],
                [2, 0, 1, 4],
            ],
            [
                [3, 0, -1, -1],
                [4, 0, -1, -1],
                [1, -1, -1, -1],
            ],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    # Set the invalid entries to a large value. This value should not be used for any purpose,
    # but setting it to a large value ensures that errors will be prodiced if it is used as an
    # index.
    indices[indices == -1] = 100

    # Create sample sizes (number of valid indices per sample)
    sample_sizes = torch.tensor([[3, 2, 4], [2, 2, 1]], dtype=torch.int64, device="cuda:0")
    indices_batch = RaggedBatch(indices, sample_sizes=sample_sizes)

    # Expected output for forward pass
    expected_output = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, -4.0, -5.0],
                [-6.0, 5.0, 6.0, -9.0, -10.0],
                [10.0, 11.0, 9.0, -14.0, 12.0],
            ],
            [
                [14.0, -17.0, -18.0, 13.0, -20.0],
                [18.0, -22.0, -23.0, -24.0, 17.0],
                [-26.0, 21.0, -28.0, -29.0, -30.0],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Expected gradients for backward pass
    if dtype.is_floating_point:
        nan = torch.nan
        expected_grad_to_write = torch.cos(
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 3.0, nan],
                        [5.0, 6.0, nan, nan],
                        [9.0, 10.0, 11.0, 12.0],
                    ],
                    [
                        [13.0, 14.0, nan, nan],
                        [17.0, 18.0, nan, nan],
                        [21.0, nan, nan, nan],
                    ],
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_grad_to_write = torch.nan_to_num(expected_grad_to_write, nan=0.0)

        expected_grad_to_fill_into = torch.cos(
            torch.tensor(
                [
                    [
                        [nan, nan, nan, -4.0, -5.0],
                        [-6.0, nan, nan, -9.0, -10.0],
                        [nan, nan, nan, -14.0, nan],
                    ],
                    [
                        [nan, -17.0, -18.0, nan, -20.0],
                        [nan, -22.0, -23.0, -24.0, nan],
                        [-26.0, nan, -28.0, -29.0, -30.0],
                    ],
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_grad_to_fill_into = torch.nan_to_num(expected_grad_to_fill_into, nan=0.0)
    else:
        expected_grad_to_write = None
        expected_grad_to_fill_into = None

    return (
        to_write_data,
        indices_batch,
        to_write_into_data,
        expected_output,
        expected_grad_to_write,
        expected_grad_to_fill_into,
    )


def _create_indexing_test_data(batch_size, num_inputs, num_outputs, additional_shape):
    """Create test data for indexing operations.

    Args:
        batch_size: Number of samples in the batch
        num_inputs: Number of input elements per sample
        num_outputs: Number of output elements per sample
        additional_shape: Additional shape dimensions

    Returns:
        Tuple of (input_data, index_batch, filler_value)
    """
    scalar_types = [torch.float, torch.float16, torch.float64, torch.bfloat16]
    filler_values = [0.0, 1.0, 2.3]

    input_shape = (batch_size, num_inputs) + additional_shape

    scalar_type = scalar_types[torch.randint(0, len(scalar_types), (1,))[0].item()]
    filler_value = filler_values[torch.randint(0, len(filler_values), (1,))[0].item()]

    input_data = torch.rand(input_shape, dtype=scalar_type, device="cuda:0")

    nums_elements = torch.randint(0, num_outputs + 1, (batch_size,), dtype=torch.int64, device="cuda:0")
    # Ensure that the maximum number of elements corresponds to the size of the `indices`
    if nums_elements.max() < num_outputs:
        to_extend_idx = torch.randint(0, batch_size, (1,)).item()
        nums_elements[to_extend_idx] = num_outputs

    # Initialize the indices with a large value to ensure that if invalid indices are used,
    # an error will be produced.
    indices = torch.full((batch_size, num_outputs), 100, dtype=torch.int64, device="cuda:0")

    for s in range(batch_size):
        ne = nums_elements[s].item()
        indices[s, 0:ne] = torch.randint(-num_inputs, num_inputs, (ne,), dtype=torch.int64, device="cuda:0")

    index_batch = RaggedBatch(indices, sample_sizes=nums_elements)

    return input_data, index_batch, filler_value


def _create_indexing_inverse_test_data(
    batch_size, num_inputs, num_outputs, additional_shape, return_to_fill_into=False
):
    """Create test data for inverste indexing operations.

    Args:
        batch_size: Number of samples in the batch
        num_inputs: Number of input elements per sample
        num_outputs: Number of output elements per sample
        additional_shape: Additional shape dimensions
        return_to_fill_into: Whether to return a second tensor (for write operations)

    Returns:
        Tuple of test data:
        - If return_to_fill_into=False: (input_data, indices_batch, filler_value)
        - If return_to_fill_into=True: (to_fill, to_fill_into, indices_batch)
    """
    scalar_types = [torch.float, torch.float16, torch.float64, torch.bfloat16]
    filler_values = [0.0, 1.0, 2.3]

    input_shape = (batch_size, num_inputs) + additional_shape
    if return_to_fill_into:
        to_fill_into_shape = (batch_size, num_outputs) + additional_shape

    scalar_type = scalar_types[torch.randint(0, len(scalar_types), (1,))[0].item()]
    filler_value = filler_values[torch.randint(0, len(filler_values), (1,))[0].item()]

    input_data = torch.rand(input_shape, dtype=scalar_type, device="cuda:0")
    if return_to_fill_into:
        to_fill_into = torch.rand(to_fill_into_shape, dtype=scalar_type, device="cuda:0")

    nums_elements = torch.randint(0, num_inputs + 1, (batch_size,), dtype=torch.int64, device="cuda:0")
    # Ensure that the maximum number of elements corresponds to the size of the `indices`
    if nums_elements.max() < num_inputs:
        to_extend_idx = torch.randint(0, batch_size, (1,)).item()
        nums_elements[to_extend_idx] = num_inputs

    # Initialize the indices with a large value to ensure that if invalid indices are used,
    # an error will be produced.
    indices = torch.full((batch_size, num_inputs), 100, dtype=torch.int64, device="cuda:0")

    for s in range(batch_size):
        ne = nums_elements[s].item()
        sampled_indices = np.random.choice(num_inputs, ne, replace=False)
        mask = np.random.random(ne) < 0.5
        sampled_indices[mask] = sampled_indices[mask] - num_inputs
        sampled_indices = torch.tensor(sampled_indices, dtype=torch.int64, device="cuda:0")
        indices[s, 0:ne] = torch.tensor(list(sampled_indices), dtype=torch.int64, device="cuda:0")

    indices_batch = RaggedBatch(indices, sample_sizes=nums_elements)

    if return_to_fill_into:
        return input_data, indices_batch, to_fill_into
    else:
        return input_data, indices_batch, filler_value


# -------------------------------------------------------------------------------------------------
# Reference implementations for testing using random data
# -------------------------------------------------------------------------------------------------


def _reference_get_values_for_indices_batched(data, indices, filler_value):
    indices_data = indices.tensor.to(dtype=torch.int64)
    nums_matches = indices.sample_sizes
    batch_size = nums_matches.shape[0]
    nums_matches_cpu = nums_matches.cpu()

    shape_to_use = list(data.shape)
    shape_to_use[1] = indices_data.shape[1]
    data_ordered = torch.ones(shape_to_use, dtype=data.dtype, device=data.device) * filler_value
    for s in range(batch_size):
        num_matches = nums_matches_cpu[s].to(dtype=torch.int64)
        data_ordered[s, 0:num_matches] = data[s, indices_data[s, 0:num_matches]]

    data_ordered = indices.create_with_sample_sizes_like_self(data_ordered)
    return data_ordered


def _reference_inverse_get_values_for_indices_batched(data, indices, num_targets_output, filler_value):
    indices_data = indices.tensor.to(dtype=torch.int64)
    batch_size = indices_data.shape[0]
    nums_matches_cpu = indices.sample_sizes.cpu()

    shape_to_use = list(data.shape)
    shape_to_use[1] = num_targets_output
    data_filled = torch.zeros(shape_to_use, dtype=data.dtype, device=data.device)
    data_filled_mask = torch.zeros(shape_to_use, dtype=torch.bool, device=data.device)
    for s in range(batch_size):
        num_matches = nums_matches_cpu[s].to(dtype=torch.int64)
        # Note that this loop is needed as the same index could be present multiple times in
        # indices[s, 0:num_matches]. In this case, the entries in the input are summed up.
        for m in range(num_matches):
            data_filled[s, indices_data[s, m]] += data[s, m]
            data_filled_mask[s, indices_data[s, m]] = True
    data_filled[torch.logical_not(data_filled_mask)] = filler_value
    return data_filled


def _reference_set_values_batched(to_set, indices, to_set_into):
    indices_data = indices.tensor.to(dtype=torch.int64)
    batch_size = indices_data.shape[0]
    nums_matches_cpu = indices.sample_sizes.cpu()

    # Avoid in-place operation on potential leaf variable
    res = to_set_into.clone()

    for s in range(batch_size):
        num_matches = nums_matches_cpu[s].to(dtype=torch.int64)
        for m in range(num_matches):
            res[s, indices_data[s, m]] = to_set[s, m]

    return res


# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_indexing_forward_manual_example(dtype, capsys):
    input_data, index_batch, filler_value, expected_output, _ = _create_manual_test_data(dtype=dtype)

    # Compute the result using the implementation
    res = batched_indexing_access(input_data, index_batch, filler_value)

    diff = res.tensor - expected_output
    max_abs_diff = torch.max(torch.abs(diff))

    # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
    # errors and the results should match exactly.
    assert max_abs_diff == 0.0, f"Difference {max_abs_diff} between implementation and reference detected"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_indexing_backward_manual_example(dtype, capsys):
    input_data, index_batch, filler_value, _, expected_grad = _create_manual_test_data(dtype=dtype)

    # Enable gradients
    input_data.requires_grad = True
    input_data.retain_grad()

    # Compute the result using the implementation
    res = batched_indexing_access(input_data, index_batch, filler_value)

    # Make sure the gradients are not identical for all elements
    res_forward_proc = torch.sin(res.tensor)
    res_sum = torch.sum(res_forward_proc)
    res_sum.backward()

    # Verify the gradients
    grad_diff = input_data.grad - expected_grad
    max_abs_diff = torch.max(torch.abs(grad_diff))

    tol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert (
        max_abs_diff < tol
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_inverse_indexing_forward_manual_example(dtype, capsys):
    filler_value = 1.4
    input_data, index_batch, expected_output, _ = _create_manual_inverse_test_data(filler_value, dtype=dtype)

    # Compute the result using the implementation
    res = batched_inverse_indexing_access(input_data, index_batch, 5, filler_value)

    diff = res - expected_output
    max_abs_diff = torch.max(torch.abs(diff))

    tol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert (
        max_abs_diff < tol
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_inverse_indexing_backward_manual_example(dtype, capsys):
    filler_value = 1.4
    input_data, index_batch, _, expected_grad = _create_manual_inverse_test_data(filler_value, dtype=dtype)

    # Enable gradients
    input_data.requires_grad = True
    input_data.retain_grad()

    # Compute the result using the implementation
    res = batched_inverse_indexing_access(input_data, index_batch, 5, 0.0)

    # Make sure the gradients are not identical for all elements
    res_forward_proc = torch.sin(res)
    res_sum = torch.sum(res_forward_proc)
    res_sum.backward()

    # Verify the gradients
    grad_diff = input_data.grad - expected_grad
    max_abs_diff = torch.max(torch.abs(grad_diff))

    # In the inverse indexing, if an index appears twice (or more times) in a single sample, the corresponding
    # values are summed up in the output. Therefore, there may be differences present due to numerical accuracy.
    # This means that in contrast to the forward indexing operation, we need to allow for small errors.
    tol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert (
        max_abs_diff < tol
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_indexing_write_forward_manual_example(dtype, capsys):
    to_write_data, indices_batch, to_write_into_data, expected_output, _, _ = _create_manual_write_test_data(
        dtype=dtype
    )

    # Compute the result using the implementation
    res = batched_indexing_write(to_write_data, indices_batch, to_write_into_data)

    diff = res - expected_output
    max_abs_diff = torch.max(torch.abs(diff))

    # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
    # errors and the results should match exactly.
    assert (
        max_abs_diff == 0.0
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_indexing_write_backward_manual_example(dtype, capsys):
    (
        to_write_data,
        indices_batch,
        to_write_into_data,
        _,
        expected_grad_to_write,
        expected_grad_to_fill_into,
    ) = _create_manual_write_test_data(dtype=dtype)

    # Enable gradients
    to_write_data.requires_grad = True
    to_write_into_data.requires_grad = True
    to_write_data.retain_grad()
    to_write_into_data.retain_grad()

    # Compute the result using the implementation
    res = batched_indexing_write(to_write_data, indices_batch, to_write_into_data)

    # Make sure the gradients are not identical for all elements
    res_forward_proc = torch.sin(res)
    res_sum = torch.sum(res_forward_proc)
    res_sum.backward()

    # Verify the gradients
    grad_diff_to_write = to_write_data.grad - expected_grad_to_write
    grad_diff_to_fill_into = to_write_into_data.grad - expected_grad_to_fill_into

    max_abs_diff_to_write = torch.max(torch.abs(grad_diff_to_write))
    max_abs_diff_to_fill_into = torch.max(torch.abs(grad_diff_to_fill_into))

    # Here, numerical errors are allowed due to the gradient computation
    tol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert (
        max_abs_diff_to_write < tol
    ), f"Difference {max_abs_diff_to_write} between implementation and reference detected for `to_write` (dtype={dtype})"
    assert (
        max_abs_diff_to_fill_into < tol
    ), f"Difference {max_abs_diff_to_fill_into} between implementation and reference detected for `to_write_into` (dtype={dtype})"


def test_indexing_forward_random_runs(capsys):
    # To see console outputs, use `with capsys.disabled(): ...
    batch_size = 12
    num_tries = 1000
    num_inputs = 50
    num_outputs = 10
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        input_data, index_batch, filler_value = _create_indexing_test_data(
            batch_size, num_inputs, num_outputs, additional_shape
        )

        ref = _reference_get_values_for_indices_batched(input_data, index_batch, filler_value)
        res = batched_indexing_access(input_data, index_batch, filler_value)

        index_diff = res.tensor - ref.tensor
        max_abs_diff = torch.max(torch.abs(index_diff)).item()

        # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
        # errors and the results should match exactly.
        assert (
            max_abs_diff == 0
        ), f"Difference of {max_abs_diff} between implementation and reference detected"


def test_indexing_backward_random_runs(capsys):
    # To see console outputs, use `with capsys.disabled(): ...`
    batch_size = 12
    num_tries = 1000
    num_inputs = 50
    num_outputs = 10
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        input_data_ref, index_batch, filler_value = _create_indexing_test_data(
            batch_size, num_inputs, num_outputs, additional_shape
        )

        # Enable gradients
        input_data_ref.requires_grad = True
        input_data_ref.retain_grad()

        with torch.no_grad():
            input_data_res = input_data_ref.clone()
        input_data_res.requires_grad = True
        input_data_res.retain_grad()

        ref_forward = _reference_get_values_for_indices_batched(input_data_ref, index_batch, filler_value)
        res_forward = batched_indexing_access(input_data_res, index_batch, filler_value)

        # Make sure the gradients are not identical for all elements
        ref_forward_proc = torch.sin(ref_forward.tensor)
        res_forward_proc = torch.sin(res_forward.tensor)

        ref_sum = torch.sum(ref_forward_proc)
        res_sum = torch.sum(res_forward_proc)

        ref_sum.backward()
        res_sum.backward()

        ref_grads = input_data_ref.grad
        res_grads = input_data_res.grad

        grad_diff = res_grads - ref_grads
        max_abs_diff = torch.max(torch.abs(grad_diff)).item()

        # The gradient computation potentially involves a summation of values (in case the same value in the input appears
        # multiple times in the output). Therefore, allow small differences due to numeric accuracy.
        assert (
            max_abs_diff < 1e-6
        ), f"Difference of {max_abs_diff} between implementation and reference detected"


def test_inverse_indexing_forward_random_runs(capsys):
    batch_size = 12
    num_tries = 1000
    num_inputs = 10
    num_outputs = 50
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        input_data, indices_batch, filler_value = _create_indexing_inverse_test_data(
            batch_size, num_inputs, num_outputs, additional_shape, False
        )

        ref = _reference_inverse_get_values_for_indices_batched(
            input_data, indices_batch, num_outputs, filler_value
        )
        res = batched_inverse_indexing_access(input_data, indices_batch, num_outputs, filler_value)

        index_diff = res - ref
        max_abs_diff = torch.max(torch.abs(index_diff)).item()

        # In the inverse indexing, if an index appears twice (or more times) in a single sample, the corresponding
        # values are summed up in the output. Therefore, there may be differences present due to numerical accuracy.
        # This means that in contrast to the forward indexing operation, we need to allow for small errors.
        assert (
            max_abs_diff < 1e-6
        ), f"Difference of {max_abs_diff} between implementation and reference detected"


def test_inverse_indexing_backward_random_runs(capsys):
    batch_size = 12
    num_tries = 100
    num_inputs = 10
    num_outputs = 50
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        input_data_ref, indices_batch, filler_value = _create_indexing_inverse_test_data(
            batch_size, num_inputs, num_outputs, additional_shape, return_to_fill_into=False
        )

        # Enable gradients
        input_data_ref.requires_grad = True
        input_data_ref.retain_grad()
        with torch.no_grad():
            input_data_res = input_data_ref.clone()
        input_data_res.requires_grad = True
        input_data_res.retain_grad()

        ref_forward = _reference_inverse_get_values_for_indices_batched(
            input_data_ref, indices_batch, num_outputs, filler_value
        )
        res_forward = batched_inverse_indexing_access(
            input_data_res, indices_batch, num_outputs, filler_value
        )

        # Make sure the gradients are not identical for all elements
        ref_forward_proc = torch.sin(ref_forward)
        res_forward_proc = torch.sin(res_forward)

        ref_sum = torch.sum(ref_forward_proc)
        res_sum = torch.sum(res_forward_proc)

        ref_sum.backward()
        res_sum.backward()

        ref_grads = input_data_ref.grad
        res_grads = input_data_res.grad

        grad_diff = res_grads - ref_grads
        max_abs_diff = torch.max(torch.abs(grad_diff)).item()

        # In the inverse indexing, if an index appears twice (or more times) in a single sample, the corresponding
        # values are summed up in the output. Therefore, there may be differences present due to numerical accuracy.
        # This means that in contrast to the forward indexing operation, we need to allow for small errors.
        assert (
            max_abs_diff < 1e-6
        ), f"Difference of {max_abs_diff} between implementation and reference detected"


def test_indexing_write_forward_random_runs(capsys):
    batch_size = 12
    num_tries = 1000
    num_to_fill = 10
    num_to_fill_into = 50
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        to_fill, indices_batch, to_fill_into = _create_indexing_inverse_test_data(
            batch_size, num_to_fill, num_to_fill_into, additional_shape, return_to_fill_into=True
        )

        res = batched_indexing_write(to_fill, indices_batch, to_fill_into.clone())
        ref = _reference_set_values_batched(to_fill, indices_batch, to_fill_into.clone())

        index_diff = res - ref
        max_abs_diff = torch.max(torch.abs(index_diff)).item()

        # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
        # errors and the results should match exactly.
        assert (
            max_abs_diff == 0.0
        ), f"Difference of {max_abs_diff} between implementation and reference detected"


def test_indexing_write_backward_random_runs(capsys):
    batch_size = 12
    num_tries = 1000
    num_to_fill = 10
    num_to_fill_into = 50
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        to_fill, indices_batch, to_fill_into = _create_indexing_inverse_test_data(
            batch_size, num_to_fill, num_to_fill_into, additional_shape, return_to_fill_into=True
        )

        to_fill_res = to_fill
        to_fill_into_res = to_fill_into
        with torch.no_grad():
            to_fill_ref = to_fill_res.clone()
            to_fill_into_ref = to_fill_into_res.clone()

        # Enable gradients
        to_fill_ref.requires_grad = True
        to_fill_into_ref.requires_grad = True
        to_fill_ref.retain_grad()
        to_fill_into_ref.retain_grad()
        to_fill_res.requires_grad = True
        to_fill_into_res.requires_grad = True
        to_fill_res.retain_grad()
        to_fill_into_res.retain_grad()

        res_forward = batched_indexing_write(to_fill_res, indices_batch, to_fill_into_res)
        ref_forward = _reference_set_values_batched(to_fill_ref, indices_batch, to_fill_into_ref)

        # Make sure the gradients are not identical for all elements
        res_forward_proc = torch.sin(res_forward)
        ref_forward_proc = torch.sin(ref_forward)

        res_sum = torch.sum(res_forward_proc)
        ref_sum = torch.sum(ref_forward_proc)

        res_sum.backward()
        ref_sum.backward()

        grad_diff_to_fill = to_fill_res.grad - to_fill_ref.grad
        grad_diff_to_fill_into = to_fill_into_res.grad - to_fill_into_ref.grad

        max_abs_diff_to_fill = torch.max(torch.abs(grad_diff_to_fill)).item()
        max_abs_diff_to_fill_into = torch.max(torch.abs(grad_diff_to_fill_into)).item()

        # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
        # errors and the results should match exactly.
        assert (
            max_abs_diff_to_fill == 0.0
        ), "Difference between implementation and reference detected for `to_fill`"
        assert (
            max_abs_diff_to_fill_into == 0.0
        ), "Difference between implementation and reference detected for `to_fill_into`"


def test_indexing_write_inconsistent_dims(capsys):
    """Test that indexing write operation fails when tensors have inconsistent number of dimensions."""
    # Create test data with different number of dimensions
    to_write_data = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32, device="cuda:0"
    )

    # Create data to write into with different number of dimensions
    to_write_into_data = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0], [-9.0, -10.0]],
            [[-11.0, -12.0], [-13.0, -14.0], [-15.0, -16.0], [-17.0, -18.0], [-19.0, -20.0]],
            [[-21.0, -22.0], [-23.0, -24.0], [-25.0, -26.0], [-27.0, -28.0], [-29.0, -30.0]],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )

    # Create indices with variable lengths per sample
    indices = torch.tensor([[1, 2, -1], [2, -1, -1], [0, 1, 4]], dtype=torch.int64, device="cuda:0")

    # Create sample sizes (number of valid indices per sample)
    sample_sizes = torch.tensor([3, 2, 4], dtype=torch.int64, device="cuda:0")
    indices_batch = RaggedBatch(indices, sample_sizes=sample_sizes)

    # Test with different number of dimensions
    with pytest.raises(RuntimeError, match=".*must have the same number of dimensions.*"):
        batched_indexing_write(to_write_data, indices_batch, to_write_into_data)


if __name__ == "__main__":
    pytest.main([__file__])
