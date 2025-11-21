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

from accvlab.batching_helpers import RaggedBatch, batched_index_mapping

# -------------------------------------------------------------------------------------------------
# Manual test data generation
# -------------------------------------------------------------------------------------------------


def _create_manual_test_data(dtype: torch.dtype = torch.float32):
    # Create input tensors
    to_fill = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Initialize to_fill_into with negative values to make original values clearly visible
    to_fill_into = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0], [-9.0, -10.0], [-11.0, -12.0]],
            [[-13.0, -14.0], [-15.0, -16.0], [-17.0, -18.0], [-19.0, -20.0], [-21.0, -22.0], [-23.0, -24.0]],
            [[-25.0, -26.0], [-27.0, -28.0], [-29.0, -30.0], [-31.0, -32.0], [-33.0, -34.0], [-35.0, -36.0]],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Create indices with different lengths for each sample
    indices_input = torch.tensor(
        [
            [0, 2, -1, -1],  # Sample 0: 2 elements
            [1, 3, 0, -1],  # Sample 1: 3 elements
            [2, 1, 0, 3],  # Sample 2: 4 elements
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    # Set the invalid entries to a large value. This value should not be used for any purpose,
    # but setting it to a large value ensures that errors will be prodiced if it is used as an
    # index.
    indices_input[indices_input == -1] = 100

    indices_output = torch.tensor(
        [
            [1, 3, -1, -1],
            [4, 2, 0, -1],
            [5, 1, 0, 2],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    indices_output[indices_output == -1] = 100

    # Expected result - original negative values should be preserved where no mapping occurs
    expected = torch.tensor(
        [
            [[-1.0, -2.0], [1.0, 2.0], [-5.0, -6.0], [5.0, 6.0], [-9.0, -10.0], [-11.0, -12.0]],  # Sample 0
            [
                [9.0, 10.0],
                [-15.0, -16.0],
                [15.0, 16.0],
                [-19.0, -20.0],
                [11.0, 12.0],
                [-23.0, -24.0],
            ],  # Sample 1
            [
                [17.0, 18.0],
                [19.0, 20.0],
                [23.0, 24.0],
                [-31.0, -32.0],
                [-33.0, -34.0],
                [21.0, 22.0],
            ],  # Sample 2
        ],
        dtype=dtype,
        device="cuda:0",
    )

    # Expected gradients (only for floating types)
    if dtype.is_floating_point:
        nan = torch.nan
        expected_to_fill_grad = torch.cos(
            torch.tensor(
                [
                    [[1.0, 2.0], [nan, nan], [5.0, 6.0], [nan, nan]],  # Sample 0
                    [[9.0, 10.0], [11.0, 12.0], [nan, nan], [15.0, 16.0]],  # Sample 1
                    [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],  # Sample 2
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_to_fill_grad = torch.nan_to_num(expected_to_fill_grad, nan=0.0)

        expected_to_fill_into_grad = torch.cos(
            torch.tensor(
                [
                    [
                        [-1.0, -2.0],
                        [nan, nan],
                        [-5.0, -6.0],
                        [nan, nan],
                        [-9.0, -10.0],
                        [-11.0, -12.0],
                    ],  # Sample 0
                    [
                        [nan, nan],
                        [-15.0, -16.0],
                        [nan, nan],
                        [-19.0, -20.0],
                        [nan, nan],
                        [-23.0, -24.0],
                    ],  # Sample 1
                    [
                        [nan, nan],
                        [nan, nan],
                        [nan, nan],
                        [-31.0, -32.0],
                        [-33.0, -34.0],
                        [nan, nan],
                    ],  # Sample 2
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_to_fill_into_grad = torch.nan_to_num(expected_to_fill_into_grad, nan=0.0)
    else:
        expected_to_fill_grad = None
        expected_to_fill_into_grad = None

    nums_elements = torch.tensor([2, 3, 4], dtype=torch.int64, device="cuda:0")

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=nums_elements)
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=nums_elements)

    return (
        to_fill,
        to_fill_into,
        indices_input_batch,
        indices_output_batch,
        expected,
        expected_to_fill_grad,
        expected_to_fill_into_grad,
    )


def _create_manual_multi_batch_dims_test_data():
    """Create test data with multiple batch dimensions.

    This creates a test case with a 2x2 batch grid (4 samples total), each with varying
    sequence lengths. The resulting tensors have shape [2, 2, max_seq_len, feature_dim],
    with appropriate sample_sizes for handling the ragged sequences.

    Returns:
        tuple: to_fill, to_fill_into, indices_input_batch, indices_output_batch
    """
    # Create input tensors with multiple batch dimensions (2x2 batch)
    # Shape: [2, 2, 3, 2] - 2 primary batches, 2 secondary batches, 3 sequence elements, 2 features
    to_fill = torch.tensor(
        [
            # First primary batch
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # First secondary batch (2 elements used)
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],  # Second secondary batch (3 elements used)
            ],
            # Second primary batch
            [
                [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],  # First secondary batch (1 element used)
                [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],  # Second secondary batch (2 elements used)
            ],
        ],
        device="cuda:0",
    )

    # Initialize to_fill_into with negative values to make original values clearly visible
    # Shape: [2, 2, 4, 2] - 2 primary batches, 2 secondary batches, 4 output sequence elements, 2 features
    to_fill_into = torch.tensor(
        [
            # First primary batch
            [
                [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]],  # First secondary batch
                [[-9.0, -10.0], [-11.0, -12.0], [-13.0, -14.0], [-15.0, -16.0]],  # Second secondary batch
            ],
            # Second primary batch
            [
                [[-17.0, -18.0], [-19.0, -20.0], [-21.0, -22.0], [-23.0, -24.0]],  # First secondary batch
                [[-25.0, -26.0], [-27.0, -28.0], [-29.0, -30.0], [-31.0, -32.0]],  # Second secondary batch
            ],
        ],
        device="cuda:0",
    )

    # Create indices with different lengths for each sample
    # Shape: [2, 2, 3] - 2 primary batches, 2 secondary batches, max 3 indices
    indices_input = torch.tensor(
        [
            # First primary batch
            [
                [0, 2, -1],  # First secondary batch: 2 elements
                [0, 1, 2],  # Second secondary batch: 3 elements
            ],
            # Second primary batch
            [
                [0, -1, -1],  # First secondary batch: 1 element
                [1, 2, -1],  # Second secondary batch: 2 elements
            ],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )

    # Set invalid entries to a large value
    indices_input[indices_input == -1] = 100

    # Shape: [2, 2, 3] - matches indices_input
    indices_output = torch.tensor(
        [
            # First primary batch
            [
                [1, 3, -1],  # First secondary batch: 2 elements
                [2, 0, 3],  # Second secondary batch: 3 elements
            ],
            # Second primary batch
            [
                [2, -1, -1],  # First secondary batch: 1 element
                [0, 3, -1],  # Second secondary batch: 2 elements
            ],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    indices_output[indices_output == -1] = 100

    # Sample sizes with multiple batch dimensions [2, 2] and inner dimension [3]
    # Shape: [2, 2] - each entry indicates number of valid elements in the corresponding sample
    sample_sizes = torch.tensor(
        [
            [2, 3],  # First primary batch
            [1, 2],  # Second primary batch
        ],
        dtype=torch.int64,
        device="cuda:0",
    )

    # Create RaggedBatch objects
    indices_input_batch = RaggedBatch(indices_input, sample_sizes=sample_sizes)
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=sample_sizes)

    # Expected result - original negative values should be preserved where no mapping occurs
    expected = torch.tensor(
        [
            # First primary batch
            [
                [[-1.0, -2.0], [1.0, 2.0], [-5.0, -6.0], [5.0, 6.0]],  # First secondary batch
                [[9.0, 10.0], [-11.0, -12.0], [7.0, 8.0], [11.0, 12.0]],  # Second secondary batch
            ],
            # Second primary batch
            [
                [[-17.0, -18.0], [-19.0, -20.0], [13.0, 14.0], [-23.0, -24.0]],  # First secondary batch
                [[21.0, 22.0], [-27.0, -28.0], [-29.0, -30.0], [23.0, 24.0]],  # Second secondary batch
            ],
        ],
        device="cuda:0",
    )

    # Expected gradients
    nan = torch.nan
    expected_to_fill_grad = torch.cos(
        torch.tensor(
            [
                # First primary batch
                [
                    [[1.0, 2.0], [nan, nan], [5.0, 6.0]],  # First secondary batch (2 elements used)
                    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],  # Second secondary batch (3 elements used)
                ],
                # Second primary batch
                [
                    [[13.0, 14.0], [nan, nan], [nan, nan]],  # First secondary batch (1 element used)
                    [[nan, nan], [21.0, 22.0], [23.0, 24.0]],  # Second secondary batch (2 elements used)
                ],
            ],
            device="cuda:0",
        )
    )
    expected_to_fill_grad = torch.nan_to_num(expected_to_fill_grad, nan=0.0)

    expected_to_fill_into_grad = torch.cos(
        torch.tensor(
            [
                # First primary batch
                [
                    [[-1.0, -2.0], [nan, nan], [-5.0, -6.0], [nan, nan]],  # First secondary batch
                    [[nan, nan], [-11.0, -12.0], [nan, nan], [nan, nan]],  # Second secondary batch
                ],
                # Second primary batch
                [
                    [[-17.0, -18.0], [-19.0, -20.0], [nan, nan], [-23.0, -24.0]],  # First secondary batch
                    [[nan, nan], [-27.0, -28.0], [-29.0, -30.0], [nan, nan]],  # Second secondary batch
                ],
            ],
            device="cuda:0",
        )
    )
    expected_to_fill_into_grad = torch.nan_to_num(expected_to_fill_into_grad, nan=0.0)

    return (
        to_fill,
        to_fill_into,
        indices_input_batch,
        indices_output_batch,
        expected,
        expected_to_fill_grad,
        expected_to_fill_into_grad,
    )


# -------------------------------------------------------------------------------------------------
# Reference implementations for testing using random data
# -------------------------------------------------------------------------------------------------


def _reference_batched_index_mapping(
    source_data: torch.Tensor,
    source_indices: RaggedBatch,
    target_indices: RaggedBatch,
    target_data: torch.Tensor,
):
    """Reference implementation of batched_index_mapping that supports multiple batch dimensions.

    This is used for testing the actual implementation against a simpler reference version.
    """
    # For multi-batch tensors, the last dimension of indices is the sequence dimension
    assert source_indices.tensor.dim() >= 2, "Indices must be at least 2D"
    assert target_indices.tensor.dim() == source_indices.tensor.dim(), "Indices dimensions must match"

    target_data_non_uniform_dim = target_indices.non_uniform_dim
    target_tensor = target_data

    # For multi-batch handling, we need to iterate over all batch dimensions
    # Get the batch shape
    batch_shape = source_indices.sample_sizes.shape
    batch_size = source_indices.sample_sizes.numel()

    # Convert indices to int64 for indexing
    source_indices_data = source_indices.tensor.to(dtype=torch.int64)
    target_indices_data = target_indices.tensor.to(dtype=torch.int64)
    nums_matches = source_indices.sample_sizes

    # Create result tensor by cloning target_tensor
    res = target_tensor.clone()

    # Iterate over all batch elements using a flattened approach
    if source_indices.num_batch_dims == 1:
        # Original single-batch case
        for i in range(batch_size):
            num_targets_curr = nums_matches[i].item()
            if num_targets_curr > 0:
                res[i][target_indices_data[i, 0:num_targets_curr]] = source_data[
                    i, source_indices_data[i, 0:num_targets_curr]
                ].squeeze()
    else:
        # For multiple batch dimensions, flatten the batch dimensions, process the single-batch-dimension case, and then reshape back.
        source_data_flat = source_data.reshape(
            source_indices.total_num_samples_in_batch, *source_data.shape[source_indices.num_batch_dims :]
        )
        target_data_flat = target_data.reshape(
            target_indices.total_num_samples_in_batch, *target_data.shape[target_indices.num_batch_dims :]
        )
        source_indices_flat = source_indices.reshape_batch_dims(source_indices.total_num_samples_in_batch)
        target_indices_flat = target_indices.reshape_batch_dims(target_indices.total_num_samples_in_batch)

        res_flat = _reference_batched_index_mapping(
            source_data_flat, source_indices_flat, target_indices_flat, target_data_flat
        )
        res = res_flat.reshape(batch_shape + res_flat.shape[1:])

    return res


# -------------------------------------------------------------------------------------------------
# Test data generation for random runs
# -------------------------------------------------------------------------------------------------


def _generate_test_data(batch_size, num_to_fill, num_to_fill_into, additional_shape):
    scalar_types = [torch.float, torch.float16, torch.float64]

    to_fill_shape = (batch_size, num_to_fill) + additional_shape
    to_fill_into_shape = (batch_size, num_to_fill_into) + additional_shape

    scalar_type = scalar_types[torch.randint(0, len(scalar_types), (1,))[0].item()]
    to_fill = torch.rand(to_fill_shape, dtype=scalar_type, device="cuda:0")
    to_fill_into = torch.rand(to_fill_into_shape, dtype=scalar_type, device="cuda:0")
    nums_elements = torch.randint(0, num_to_fill + 1, (batch_size,), dtype=torch.int64, device="cuda:0")
    # Initialize the indices with a large value to ensure that if invalid indices are used,
    # an error will be produced.
    indices_input = torch.full((batch_size, num_to_fill), 100, dtype=torch.int64, device="cuda:0")
    indices_output = torch.full((batch_size, num_to_fill), 100, dtype=torch.int64, device="cuda:0")

    # Ensure that the maximum number of elements corresponds to the size of the `indices_input`
    # & `indices_output`
    if nums_elements.max() < num_to_fill:
        to_extend_idx = torch.randint(0, batch_size, (1,)).item()
        nums_elements[to_extend_idx] = num_to_fill

    for s in range(batch_size):
        use_negative_input = torch.randint(0, 2, (1,)).item() > 0
        use_negative_output = torch.randint(0, 2, (1,)).item() > 0
        ne = nums_elements[s].item()
        if use_negative_input:
            index_range_input = range(-num_to_fill, 0)
        else:
            index_range_input = range(0, num_to_fill)
        if use_negative_output:
            index_range_output = range(-num_to_fill_into, 0)
        else:
            index_range_output = range(0, num_to_fill_into)
        sampled_indices_input = np.random.choice(index_range_input, ne, replace=False)
        sampled_indices_output = np.random.choice(index_range_output, ne, replace=False)
        indices_input[s, 0:ne] = torch.tensor(sampled_indices_input)
        indices_output[s, 0:ne] = torch.tensor(sampled_indices_output)

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=nums_elements)
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=nums_elements)

    return to_fill, to_fill_into, indices_input_batch, indices_output_batch


def _generate_multi_batch_dim_test_data(batch_shape, num_to_fill, num_to_fill_into, additional_shape):
    """Generate random test data with multiple batch dimensions.

    Args:
        batch_shape (tuple): Shape of the batch dimensions (e.g., (2, 3) for 2x3 batch grid)
        num_to_fill (int): Maximum number of elements in each sample for the source tensor
        num_to_fill_into (int): Maximum number of elements in each sample for the target tensor
        additional_shape (tuple): Shape of each element (e.g., (2, 3) for 2x3 features)

    Returns:
        tuple: to_fill, to_fill_into, indices_input_batch, indices_output_batch
    """
    scalar_types = [torch.float, torch.float16, torch.float64]

    # Create tensor shapes with multiple batch dimensions
    to_fill_shape = batch_shape + (num_to_fill,) + additional_shape
    to_fill_into_shape = batch_shape + (num_to_fill_into,) + additional_shape

    # Random scalar type
    scalar_type = scalar_types[torch.randint(0, len(scalar_types), (1,))[0].item()]

    # Create data tensors
    to_fill = torch.rand(to_fill_shape, dtype=scalar_type, device="cuda:0")
    to_fill_into = torch.rand(to_fill_into_shape, dtype=scalar_type, device="cuda:0")

    # Create sample sizes tensor matching the batch shape
    nums_elements = torch.randint(1, num_to_fill + 1, batch_shape, dtype=torch.int64, device="cuda:0")

    # Initialize indices tensors
    indices_shape = batch_shape + (num_to_fill,)
    indices_input = torch.full(indices_shape, 100, dtype=torch.int64, device="cuda:0")
    indices_output = torch.full(indices_shape, 100, dtype=torch.int64, device="cuda:0")

    # Fill in indices for each sample in the batch
    # We need to iterate over all combinations of batch indices
    batch_indices = torch.meshgrid(*[torch.arange(dim) for dim in batch_shape], indexing='ij')
    for sample_idx in zip(*[idx.flatten() for idx in batch_indices]):
        # Create a tuple of indices for accessing the current sample
        idx_tuple = sample_idx

        # Get the number of elements for this sample
        ne = nums_elements[idx_tuple].item()

        # Randomly choose whether to use negative indices
        use_negative_input = torch.randint(0, 2, (1,)).item() > 0
        use_negative_output = torch.randint(0, 2, (1,)).item() > 0

        # Define index ranges
        if use_negative_input:
            index_range_input = range(-num_to_fill, 0)
        else:
            index_range_input = range(0, num_to_fill)
        if use_negative_output:
            index_range_output = range(-num_to_fill_into, 0)
        else:
            index_range_output = range(0, num_to_fill_into)

        # Sample random indices
        sampled_indices_input = np.random.choice(index_range_input, ne, replace=False)
        sampled_indices_output = np.random.choice(index_range_output, ne, replace=False)

        # Assign indices to the tensors
        # We need to add a slice for the sequence dimension
        indices_input[idx_tuple + (slice(0, ne),)] = torch.tensor(sampled_indices_input, device="cuda:0")
        indices_output[idx_tuple + (slice(0, ne),)] = torch.tensor(sampled_indices_output, device="cuda:0")

    # Create RaggedBatch objects
    indices_input_batch = RaggedBatch(indices_input, sample_sizes=nums_elements)
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=nums_elements)

    return to_fill, to_fill_into, indices_input_batch, indices_output_batch


# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_batched_index_mapping_forward_manual_example(dtype, capsys):
    to_fill, to_fill_into, indices_input_batch, indices_output_batch, expected, _, _ = (
        _create_manual_test_data(dtype=dtype)
    )

    # Forward pass test
    res = batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into.clone())

    # Verify forward pass
    diff = res - expected
    max_abs_diff = torch.max(torch.abs(diff)).item()
    assert max_abs_diff == 0.0, "Difference between implementation and reference detected for `to_fill`"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_batched_index_mapping_backward_manual_example(dtype, capsys):
    (
        to_fill,
        to_fill_into,
        indices_input_batch,
        indices_output_batch,
        _,
        expected_to_fill_grad,
        expected_to_fill_into_grad,
    ) = _create_manual_test_data(dtype=dtype)

    # Enable gradients for both input tensors
    to_fill.requires_grad = True
    to_fill_into.requires_grad = True

    # Compute result with gradients enabled
    res = batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)

    # Apply a non-linear transformation to make gradients non-trivial
    # We use sin(x) which has gradient cos(x) at each point
    res_proc = torch.sin(res)

    # Compute sum to get a scalar for backward pass
    res_sum = torch.sum(res_proc)

    # Compute gradients
    res_sum.backward()

    # Verify gradients
    atol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert torch.allclose(
        to_fill.grad, expected_to_fill_grad, atol=atol, rtol=0
    ), "Gradients for to_fill do not match expected values"
    assert torch.allclose(
        to_fill_into.grad, expected_to_fill_into_grad, atol=atol, rtol=0
    ), "Gradients for to_fill_into do not match expected values"


def test_batched_index_mapping_multiple_batch_dims_forward_dims_manual_example(capsys):
    """Test that batched_index_mapping works correctly with multiple batch dimensions."""
    # Get the test data with multiple batch dimensions
    to_fill, to_fill_into, indices_input_batch, indices_output_batch, expected, _, _ = (
        _create_manual_multi_batch_dims_test_data()
    )

    # Run the batched_index_mapping operation
    result = batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into.clone())

    # Compare results between implementation and reference
    diff = result - expected
    max_abs_diff = torch.max(torch.abs(diff)).item()

    # Debug output for differences between implementations
    if max_abs_diff > 1e-5:
        diff_locations = torch.where(torch.abs(diff) > 1e-5)
        print("\nDifferences between implementation and reference:")
        for i in range(min(5, len(diff_locations[0]))):  # Show up to 5 differences
            idx = tuple(d[i] for d in diff_locations)
            print(f"Location {idx}:")
            print(f"  Implementation: {result[idx].item()}")
            print(f"  Reference:      {expected[idx].item()}")
            print(f"  Diff:           {diff[idx].item()}")

    # The implementations should produce the same result
    assert max_abs_diff == 0.0, f"Difference between implementation and reference detected: {max_abs_diff}"


def test_batched_index_mapping_multiple_batch_dims_dims_backward_manual_example(capsys):
    """Test that gradients flow correctly with multiple batch dimensions."""
    # Get the multi-batch test data
    (
        to_fill,
        to_fill_into,
        indices_input_batch,
        indices_output_batch,
        _,
        expected_to_fill_grad,
        expected_to_fill_into_grad,
    ) = _create_manual_multi_batch_dims_test_data()

    # Enable gradients for input tensors
    to_fill.requires_grad = True
    to_fill_into.requires_grad = True

    # Compute results with gradients enabled
    res = batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)

    # Apply a non-linear transformation to make gradients non-trivial
    res_proc = torch.sin(res)

    # Compute sum for backward pass
    res_sum = torch.sum(res_proc)

    # Compute gradients
    res_sum.backward()

    # Verify gradients match between implementation and reference
    grad_diff_to_fill = to_fill.grad - expected_to_fill_grad
    grad_diff_to_fill_into = to_fill_into.grad - expected_to_fill_into_grad

    max_abs_diff_to_fill = torch.max(torch.abs(grad_diff_to_fill)).item()
    max_abs_diff_to_fill_into = torch.max(torch.abs(grad_diff_to_fill_into)).item()

    # The gradients should match exactly
    assert max_abs_diff_to_fill == 0.0, f"Gradient difference for to_fill detected: {max_abs_diff_to_fill}"
    assert (
        max_abs_diff_to_fill_into == 0.0
    ), f"Gradient difference for to_fill_into detected: {max_abs_diff_to_fill_into}"


def test_batched_index_mapping_forward_random_runs(capsys):
    batch_size = 12
    num_tries = 1000
    num_to_fill = 10
    num_to_fill_into = 50
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        to_fill, to_fill_into, indices_input_batch, indices_output_batch = _generate_test_data(
            batch_size, num_to_fill, num_to_fill_into, additional_shape
        )

        res = batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into.clone())
        ref = _reference_batched_index_mapping(
            to_fill, indices_input_batch, indices_output_batch, to_fill_into.clone()
        )

        index_diff = res - ref
        max_abs_diff = torch.max(torch.abs(index_diff)).item()

        # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
        # errors and the results should match exactly.
        assert max_abs_diff == 0.0, "Difference between implementation and reference detected"


def test_batched_index_mapping_backward_random_runs(capsys):
    batch_size = 12
    num_tries = 1000
    num_to_fill = 10
    num_to_fill_into = 50
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        to_fill, to_fill_into, indices_input_batch, indices_output_batch = _generate_test_data(
            batch_size, num_to_fill, num_to_fill_into, additional_shape
        )

        to_fill_res = to_fill
        to_fill_into_res = to_fill_into
        with torch.no_grad():
            to_fill_ref = to_fill.clone()
            to_fill_into_ref = to_fill_into.clone()
        to_fill_res.requires_grad = True
        to_fill_into_res.requires_grad = True
        to_fill_ref.requires_grad = True
        to_fill_into_ref.requires_grad = True
        to_fill_res.retain_grad()
        to_fill_into_res.retain_grad()
        to_fill_ref.retain_grad()
        to_fill_into_ref.retain_grad()

        res_forward = batched_index_mapping(
            to_fill_res, indices_input_batch, indices_output_batch, to_fill_into_res
        )
        ref_forward = _reference_batched_index_mapping(
            to_fill_ref, indices_input_batch, indices_output_batch, to_fill_into_ref
        )

        # Make sure the gradients for different elements are different (to detect errors using the wrong gradients)
        res_proc = torch.sin(res_forward)
        ref_proc = torch.sin(ref_forward)

        # Start the backward step
        res_sum = torch.sum(res_proc)
        ref_sum = torch.sum(ref_proc)
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


def test_batched_index_mapping_multi_batch_dims_forward_random_runs(capsys):
    """Test batched_index_mapping with random data and multiple batch dimensions."""
    batch_shape = (2, 3)  # 2x3 batch grid
    num_tries = 50
    num_to_fill = 8
    num_to_fill_into = 12
    additional_shape = (2, 3)  # 2x3 features

    for i in range(num_tries):
        # Generate random multi-batch test data
        to_fill, to_fill_into, indices_input_batch, indices_output_batch = (
            _generate_multi_batch_dim_test_data(batch_shape, num_to_fill, num_to_fill_into, additional_shape)
        )

        # Run both the implementation and reference
        res = batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into.clone())
        ref = _reference_batched_index_mapping(
            to_fill, indices_input_batch, indices_output_batch, to_fill_into.clone()
        )

        # Compare results
        index_diff = res - ref
        max_abs_diff = torch.max(torch.abs(index_diff)).item()

        # The results should match exactly
        assert (
            max_abs_diff == 0.0
        ), f"Difference between implementation and reference detected in random trial {i}: {max_abs_diff}"


def test_batched_index_mapping_multi_batch_dims_backward_random_runs(capsys):
    """Test gradients for batched_index_mapping with random data and multiple batch dimensions."""
    batch_shape = (2, 3)  # 2x3 batch grid
    num_tries = 20  # Fewer tries as this is more expensive
    num_to_fill = 8
    num_to_fill_into = 12
    additional_shape = (2, 3)  # 2x3 features

    for i in range(num_tries):
        # Generate random multi-batch test data
        to_fill, to_fill_into, indices_input_batch, indices_output_batch = (
            _generate_multi_batch_dim_test_data(batch_shape, num_to_fill, num_to_fill_into, additional_shape)
        )

        # Enable gradients for both implementation and reference
        to_fill_res = to_fill.clone()
        to_fill_into_res = to_fill_into.clone()
        with torch.no_grad():
            to_fill_ref = to_fill.clone()
            to_fill_into_ref = to_fill_into.clone()

        to_fill_res.requires_grad = True
        to_fill_into_res.requires_grad = True
        to_fill_ref.requires_grad = True
        to_fill_into_ref.requires_grad = True

        to_fill_res.retain_grad()
        to_fill_into_res.retain_grad()
        to_fill_ref.retain_grad()
        to_fill_into_ref.retain_grad()

        # Compute forward results
        res_forward = batched_index_mapping(
            to_fill_res, indices_input_batch, indices_output_batch, to_fill_into_res
        )
        ref_forward = _reference_batched_index_mapping(
            to_fill_ref, indices_input_batch, indices_output_batch, to_fill_into_ref
        )

        # Make sure the gradients for different elements are different (to detect errors using the wrong gradients)
        res_proc = torch.sin(res_forward)
        ref_proc = torch.sin(ref_forward)

        # Start the backward step
        res_sum = torch.sum(res_proc)
        ref_sum = torch.sum(ref_proc)
        res_sum.backward()
        ref_sum.backward()

        # Compare gradients
        grad_diff_to_fill = to_fill_res.grad - to_fill_ref.grad
        grad_diff_to_fill_into = to_fill_into_res.grad - to_fill_into_ref.grad

        max_abs_diff_to_fill = torch.max(torch.abs(grad_diff_to_fill)).item()
        max_abs_diff_to_fill_into = torch.max(torch.abs(grad_diff_to_fill_into)).item()

        # The gradients should match exactly
        assert (
            max_abs_diff_to_fill == 0.0
        ), f"Trial {i}: Gradient difference for to_fill detected: {max_abs_diff_to_fill}"
        assert (
            max_abs_diff_to_fill_into == 0.0
        ), f"Trial {i}: Gradient difference for to_fill_into detected: {max_abs_diff_to_fill_into}"


def test_batched_index_mapping_mismatched_indices(capsys):
    # Create test data with mismatched sizes
    to_fill = torch.randn(2, 5, device='cuda:0')
    to_fill_into = torch.randn(2, 5, device='cuda:0')

    # Create indices with different sample sizes
    indices_input = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')
    indices_output = torch.tensor([[0, 1, -1], [0, 1, 2]], device='cuda:0')  # Different sizes than input

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=torch.tensor([3, 4], device='cuda:0'))
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=torch.tensor([2, 3], device='cuda:0'))

    # Expect an error when calling the function
    with pytest.raises(AssertionError):
        batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)


def test_batched_index_mapping_mismatched_index_types(capsys):
    # Create test data
    to_fill = torch.randn(2, 5, device='cuda:0')
    to_fill_into = torch.randn(2, 5, device='cuda:0')

    # Create indices with different types
    indices_input = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0', dtype=torch.int64)
    indices_output = torch.tensor(
        [[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0', dtype=torch.int32
    )  # Different type than input

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=torch.tensor([3, 4], device='cuda:0'))
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=torch.tensor([3, 4], device='cuda:0'))

    # Expect an error when calling the function
    with pytest.raises(RuntimeError):
        batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)


def test_batched_index_mapping_mismatched_data_types(capsys):
    # Create test data
    to_fill = torch.randn(2, 5, device='cuda:0', dtype=torch.float64)
    to_fill_into = torch.randn(2, 5, device='cuda:0', dtype=torch.float32)

    # Create indices with different types
    indices_input = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')
    indices_output = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=torch.tensor([3, 4], device='cuda:0'))
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=torch.tensor([3, 4], device='cuda:0'))

    # Expect an error when calling the function
    with pytest.raises(RuntimeError):
        batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)


def test_batched_index_mapping_mismatched_data_sizes(capsys):
    # Create test data
    to_fill = torch.randn(2, 5, device='cuda:0')
    to_fill_into = torch.randn(2, 5, 3, device='cuda:0')

    # Create indices with different types
    indices_input = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')
    indices_output = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=torch.tensor([3, 4], device='cuda:0'))
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=torch.tensor([3, 4], device='cuda:0'))

    # Expect an error when calling the function
    with pytest.raises(RuntimeError):
        batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)


def test_batched_index_mapping_mismatched_devices(capsys):
    # Create test data
    to_fill = torch.randn(2, 5, device='cuda:0')
    to_fill_into = torch.randn(2, 5, 3, device='cpu')

    # Create indices with different types
    indices_input = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')
    indices_output = torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], device='cuda:0')

    indices_input_batch = RaggedBatch(indices_input, sample_sizes=torch.tensor([3, 4], device='cuda:0'))
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=torch.tensor([3, 4], device='cuda:0'))

    # Expect an error when calling the function
    with pytest.raises(RuntimeError):
        batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)


def test_batched_index_mapping_indices_wrong_num_dims(capsys):
    # Create test data
    to_fill = torch.randn(2, 5, device='cuda:0')
    to_fill_into = torch.randn(2, 5, 3, device='cuda:0')

    # Create indices with different types
    indices_input = torch.tensor([[[0, 1, 2, -1]], [[0, 1, 2, 3]]], device='cuda:0')
    indices_output = torch.tensor([[[0, 1, 2, -1]], [[0, 1, 2, 3]]], device='cuda:0')
    indices_input_batch = RaggedBatch(indices_input, sample_sizes=torch.tensor([3, 4], device='cuda:0'))
    indices_output_batch = RaggedBatch(indices_output, sample_sizes=torch.tensor([3, 4], device='cuda:0'))

    # Expect an error when calling the function
    with pytest.raises(AssertionError):
        batched_index_mapping(to_fill, indices_input_batch, indices_output_batch, to_fill_into)


def test_batched_index_mapping_multi_batch_random_backward_runs(capsys):
    """Test gradients with randomly generated multi-batch data."""
    batch_shape = (2, 3)  # 2x3 batch grid
    num_tries = 20  # Fewer tries as this is more expensive
    num_to_fill = 8
    num_to_fill_into = 12
    additional_shape = (2, 3)  # 2x3 features

    for i in range(num_tries):
        # Generate random multi-batch test data
        to_fill, to_fill_into, indices_input_batch, indices_output_batch = (
            _generate_multi_batch_dim_test_data(batch_shape, num_to_fill, num_to_fill_into, additional_shape)
        )

        # Clone and enable gradients
        to_fill_res = to_fill.clone().detach().requires_grad_(True)
        to_fill_into_res = to_fill_into.clone().detach().requires_grad_(True)
        to_fill_ref = to_fill.clone().detach().requires_grad_(True)
        to_fill_into_ref = to_fill_into.clone().detach().requires_grad_(True)

        # Compute forward results
        res_forward = batched_index_mapping(
            to_fill_res, indices_input_batch, indices_output_batch, to_fill_into_res
        )
        ref_forward = _reference_batched_index_mapping(
            to_fill_ref, indices_input_batch, indices_output_batch, to_fill_into_ref
        )

        # Apply non-linear transformation
        res_proc = torch.sin(res_forward)
        ref_proc = torch.sin(ref_forward)

        # Compute backward gradients
        res_sum = torch.sum(res_proc)
        ref_sum = torch.sum(ref_proc)
        res_sum.backward()
        ref_sum.backward()

        # Compare gradients
        grad_diff_to_fill = to_fill_res.grad - to_fill_ref.grad
        grad_diff_to_fill_into = to_fill_into_res.grad - to_fill_into_ref.grad

        max_abs_diff_to_fill = torch.max(torch.abs(grad_diff_to_fill)).item()
        max_abs_diff_to_fill_into = torch.max(torch.abs(grad_diff_to_fill_into)).item()

        assert (
            max_abs_diff_to_fill == 0.0
        ), f"Trial {i}: Gradient difference for to_fill detected: {max_abs_diff_to_fill}"
        assert (
            max_abs_diff_to_fill_into == 0.0
        ), f"Trial {i}: Gradient difference for to_fill_into detected: {max_abs_diff_to_fill_into}"


if __name__ == "__main__":
    pytest.main([__file__])
