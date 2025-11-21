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

import sys
import os

from accvlab.batching_helpers import (
    RaggedBatch,
    batched_bool_indexing,
    batched_bool_indexing_write,
)

# -------------------------------------------------------------------------------------------------
# Test data generation helpers
# -------------------------------------------------------------------------------------------------


def _create_simple_ragged_batch_test_data(dtype: torch.dtype = torch.float32):
    """Create test data for simple RaggedBatch input (one batch dimension, non-uniform dim is 1)."""
    batch_size = 4
    max_sample_size = 5
    additional_dims = (3, 2)  # Additional dimensions after the non-uniform dimension

    # Create input data with distinct values for easy verification
    input_data = torch.arange(
        batch_size * max_sample_size * np.prod(additional_dims), dtype=dtype, device="cuda:0"
    )
    input_data = input_data.reshape(batch_size, max_sample_size, *additional_dims)

    # Create sample sizes (different sizes for each sample)
    sample_sizes = torch.tensor([3, 5, 2, 4], dtype=torch.int64, device="cuda:0")

    # Create mask based on sample sizes
    mask = torch.zeros((batch_size, max_sample_size), dtype=torch.bool, device="cuda:0")
    for i, size in enumerate(sample_sizes):
        mask[i, :size] = True

    # Create boolean mask for indexing (select some elements from each sample)
    input_mask = torch.zeros((batch_size, max_sample_size), dtype=torch.bool, device="cuda:0")
    input_mask[0, [0, 2, 4]] = (
        True  # Sample 0: select indices 0, 2 (note that index 4 corresponds to a filler value)
    )
    input_mask[1, [1, 3, 4]] = True  # Sample 1: select indices 1, 3, 4
    input_mask[2, [0, 3]] = True  # Sample 2: select index 0 (note that index 3 corresponds to a filler value)
    input_mask[3, [0, 1, 2, 3, 4]] = (
        True  # Sample 3: select all indices (note that index 4 corresponds to a filler value)
    )

    # Create expected output
    expected_sample_sizes = torch.tensor([2, 3, 1, 4], dtype=torch.int64, device="cuda:0")
    max_output_size = 4
    expected_output = torch.full(
        (batch_size, max_output_size, *additional_dims), fill_value=0.0, dtype=dtype, device="cuda:0"
    )

    # Fill expected output with selected values
    expected_output[0, 0] = input_data[0, 0]  # Sample 0, first selected element
    expected_output[0, 1] = input_data[0, 2]  # Sample 0, second selected element

    expected_output[1, 0] = input_data[1, 1]  # Sample 1, first selected element
    expected_output[1, 1] = input_data[1, 3]  # Sample 1, second selected element
    expected_output[1, 2] = input_data[1, 4]  # Sample 1, third selected element

    expected_output[2, 0] = input_data[2, 0]  # Sample 2, first selected element

    expected_output[3, 0] = input_data[3, 0]  # Sample 3, first selected element
    expected_output[3, 1] = input_data[3, 1]  # Sample 3, second selected element
    expected_output[3, 2] = input_data[3, 2]  # Sample 3, third selected element
    expected_output[3, 3] = input_data[3, 3]  # Sample 3, fourth selected element

    # Create RaggedBatch instances
    input_ragged = RaggedBatch(input_data, sample_sizes=sample_sizes)
    input_mask_ragged = RaggedBatch(input_mask, sample_sizes=sample_sizes)

    return input_ragged, input_mask_ragged, expected_output, expected_sample_sizes


def _create_complex_multi_dim_test_data(dtype: torch.dtype = torch.float32):
    """Create test data for complex multi-dimensional RaggedBatch input."""
    batch_shape = (2, 3)  # Multi-dimensional batch
    max_sample_size = 4
    additional_dims = (2, 3)  # Additional dimensions after the non-uniform dimension

    # Create input data
    total_batch_size = np.prod(batch_shape)
    input_data = torch.arange(
        total_batch_size * max_sample_size * np.prod(additional_dims), dtype=dtype, device="cuda:0"
    )
    input_data = input_data.reshape(*batch_shape, max_sample_size, *additional_dims)

    # Create sample sizes (different sizes for each sample)
    sample_sizes = torch.tensor([[3, 4, 2], [1, 3, 4]], dtype=torch.int64, device="cuda:0")

    # Create mask based on sample sizes
    mask = torch.zeros((*batch_shape, max_sample_size), dtype=torch.bool, device="cuda:0")
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            mask[i, j, : sample_sizes[i, j]] = True

    # Create boolean mask for indexing (select some elements from each sample)
    # Since input_data is a multi-dimensional RaggedBatch, the mask must also be a RaggedBatch
    input_mask = torch.zeros((*batch_shape, max_sample_size), dtype=torch.bool, device="cuda:0")
    input_mask[0, 0, [0, 2, 3]] = (
        True  # Sample (0,0): select indices 0, 2 (note that index 3 corresponds to a filler value)
    )
    input_mask[0, 1, [1, 2, 3]] = True  # Sample (0,1): select indices 1, 2, 3
    input_mask[0, 2, [0, 2]] = (
        True  # Sample (0,2): select index 0 (note that index 2 corresponds to a filler value)
    )
    input_mask[1, 0, [0]] = True  # Sample (1,0): select index 0
    input_mask[1, 1, [0, 2]] = True  # Sample (1,1): select indices 0, 2
    input_mask[1, 2, [1, 3]] = True  # Sample (1,2): select indices 1, 3

    # Create expected output
    expected_sample_sizes = torch.tensor([[2, 3, 1], [1, 2, 2]], dtype=torch.int64, device="cuda:0")
    max_output_size = 3
    expected_output = torch.full(
        (*batch_shape, max_output_size, *additional_dims),
        fill_value=0.0,
        dtype=dtype,
        device="cuda:0",
    )

    # Fill expected output with selected values
    expected_output[0, 0, 0] = input_data[0, 0, 0]  # Sample (0,0), first selected element
    expected_output[0, 0, 1] = input_data[0, 0, 2]  # Sample (0,0), second selected element

    expected_output[0, 1, 0] = input_data[0, 1, 1]  # Sample (0,1), first selected element
    expected_output[0, 1, 1] = input_data[0, 1, 2]  # Sample (0,1), second selected element
    expected_output[0, 1, 2] = input_data[0, 1, 3]  # Sample (0,1), third selected element

    expected_output[0, 2, 0] = input_data[0, 2, 0]  # Sample (0,2), first selected element

    expected_output[1, 0, 0] = input_data[1, 0, 0]  # Sample (1,0), first selected element

    expected_output[1, 1, 0] = input_data[1, 1, 0]  # Sample (1,1), first selected element
    expected_output[1, 1, 1] = input_data[1, 1, 2]  # Sample (1,1), second selected element

    expected_output[1, 2, 0] = input_data[1, 2, 1]  # Sample (1,2), first selected element
    expected_output[1, 2, 1] = input_data[1, 2, 3]  # Sample (1,2), second selected element

    # Create RaggedBatch instances
    input_ragged = RaggedBatch(input_data, sample_sizes=sample_sizes)
    input_mask_ragged = RaggedBatch(input_mask, sample_sizes=sample_sizes)

    return input_ragged, input_mask_ragged, expected_output, expected_sample_sizes


def _create_simple_inverse_indexing_test_data(dtype: torch.dtype = torch.float32):
    """Create test data for simple inverse indexing (batched_bool_indexing_write)."""
    # Reuse the simple test data but swap the roles
    input_ragged, input_mask_ragged, indexed_output, indexed_sample_sizes = (
        _create_simple_ragged_batch_test_data(dtype=dtype)
    )

    # For inverse indexing:
    # - to_write = indexed_output (the result of the forward indexing)
    # - mask_for_output = input_mask_ragged (the original mask)
    # - to_write_into = modified version of input_ragged with wrong values at masked positions
    # - expected_output = original input_ragged (the correct values)

    to_write = RaggedBatch(indexed_output, sample_sizes=indexed_sample_sizes)
    mask_for_output = input_mask_ragged

    # Create to_write_into by copying the original input but setting wrong values at masked positions
    to_write_into_tensor = input_ragged.tensor.clone()
    for i in range(input_ragged.tensor.shape[0]):
        valid_size = input_ragged.sample_sizes[i]
        mask_indices = torch.where(input_mask_ragged.tensor[i, :valid_size])[0]
        for mask_idx in mask_indices:
            # Set wrong values at masked positions so we can verify writing occurred
            if input_ragged.tensor.dtype.is_floating_point:
                fill_val = torch.nan
            else:
                fill_val = -999
            to_write_into_tensor[i, mask_idx] = fill_val
    # Create RaggedBatch instance with the correct sample sizes
    to_write_into = input_ragged.create_with_sample_sizes_like_self(to_write_into_tensor)

    # The expected output should be the original input data
    expected_output = input_ragged.tensor

    return to_write, mask_for_output, to_write_into, expected_output


def _create_complex_multi_dim_inverse_indexing_test_data(dtype: torch.dtype = torch.float32):
    """Create test data for complex multi-dimensional inverse indexing."""
    # Reuse the complex test data but swap the roles
    input_ragged, input_mask_ragged, indexed_output, indexed_sample_sizes = (
        _create_complex_multi_dim_test_data(dtype=dtype)
    )

    # For inverse indexing:
    # - to_write = indexed_output (the result of the forward indexing)
    # - mask_for_output = input_mask_ragged (the original mask)
    # - to_write_into = modified version of input_ragged with wrong values at masked positions
    # - expected_output = original input_ragged (the correct values)

    to_write = RaggedBatch(indexed_output, sample_sizes=indexed_sample_sizes)
    mask_for_output = input_mask_ragged

    # Create to_write_into by copying the original input but setting wrong values at masked positions
    to_write_into_tensor = input_ragged.tensor.clone()
    batch_shape = input_ragged.batch_shape
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            valid_size = input_ragged.sample_sizes[i, j]
            mask_indices = torch.where(input_mask_ragged.tensor[i, j, :valid_size])[0]
            for mask_idx in mask_indices:
                # Set wrong values at masked positions so we can verify writing occurred
                if input_ragged.tensor.dtype.is_floating_point:
                    fill_val = torch.nan
                else:
                    fill_val = -999
                to_write_into_tensor[i, j, mask_idx] = fill_val
    # Create RaggedBatch instance with the correct sample sizes
    to_write_into = input_ragged.create_with_sample_sizes_like_self(to_write_into_tensor)

    # The expected output should be the original input data
    expected_output = input_ragged.tensor

    return to_write, mask_for_output, to_write_into, expected_output


# -------------------------------------------------------------------------------------------------
# Test functions
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_simple_ragged_batch_input(dtype):
    """Test batched_boolean_indexing with simple RaggedBatch input."""
    input_ragged, input_mask_ragged, expected_output, expected_sample_sizes = (
        _create_simple_ragged_batch_test_data(dtype=dtype)
    )

    # Apply boolean indexing
    result = batched_bool_indexing(input_ragged, input_mask_ragged)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check sample sizes
    assert torch.all(result.sample_sizes == expected_sample_sizes)

    # Check output tensor shape
    assert result.tensor.shape == expected_output.shape

    # Check output values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_complex_multi_dim_ragged_batch_input(dtype):
    """Test batched_boolean_indexing with complex multi-dimensional RaggedBatch input (both inputs are RaggedBatch)."""
    input_ragged, input_mask_ragged, expected_output, expected_sample_sizes = (
        _create_complex_multi_dim_test_data(dtype=dtype)
    )
    # Apply boolean indexing
    result = batched_bool_indexing(input_ragged, input_mask_ragged)
    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)
    # Check sample sizes
    assert torch.all(result.sample_sizes == expected_sample_sizes)
    # Check output tensor shape
    assert result.tensor.shape[0] == expected_output.shape[0]  # first batch dim
    assert result.tensor.shape[1] == expected_output.shape[1]  # second batch dim
    assert result.tensor.shape[2] == expected_output.shape[2]  # max sample size
    assert result.tensor.shape[3:] == expected_output.shape[3:]  # additional dimensions
    # Check output values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_complex_multi_dim_ragged_batch_input_transposed(dtype):
    """Test batched_boolean_indexing with complex multi-dimensional RaggedBatch input where the non-uniform dimension is transposed."""
    input_ragged, input_mask_ragged, expected_output, expected_sample_sizes = (
        _create_complex_multi_dim_test_data(dtype=dtype)
    )

    # Transpose the non-uniform dimension to the last dimension (position 4)
    # Original shape: (2, 3, 4, 2, 3) with non_uniform_dim=2
    # Transposed shape: (2, 3, 2, 3, 4) with non_uniform_dim=4
    input_ragged_transposed = input_ragged.get_non_uniform_dimension_transposed_to(4)

    # Apply boolean indexing
    result = batched_bool_indexing(input_ragged_transposed, input_mask_ragged)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check sample sizes (should remain the same)
    assert torch.all(result.sample_sizes == expected_sample_sizes)

    # Check that the non-uniform dimension is preserved in the result
    assert result.non_uniform_dim == 4

    # Check output tensor shape
    # The result should preserve the same dimension structure as the transposed input
    # Transposed input: (2, 3, 3, 2, 4) with non_uniform_dim=4
    # Expected result: (2, 3, 3, 2, 3) with non_uniform_dim=4 (max sample size after indexing)
    assert result.tensor.shape[0] == 2  # first batch dim
    assert result.tensor.shape[1] == 3  # second batch dim
    assert result.tensor.shape[2] == 3  # first additional dim
    assert result.tensor.shape[3] == 2  # second additional dim
    assert result.tensor.shape[4] == 3  # max sample size (non-uniform dim)

    # Transpose the result back to the original non_uniform_dim (position 2)
    # This ensures that the result has the same structure as the expected output
    result_transposed_back = result.get_non_uniform_dimension_transposed_to(2)

    # Check that the transposed result matches the expected output
    assert torch.equal(result_transposed_back.tensor, expected_output)

    # Check that the sample sizes are preserved
    assert torch.all(result_transposed_back.sample_sizes == expected_sample_sizes)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_ragged_batch_tensor_input_combinations(dtype):
    """Test batched_boolean_indexing when only one input is a RaggedBatch (with a single batch dimension)."""
    # Use data from simple test
    input_ragged, input_mask_ragged, expected_output, expected_sample_sizes = (
        _create_simple_ragged_batch_test_data(dtype=dtype)
    )

    # ----- Test with tensor mask, RaggedBatch input -----
    # Apply boolean indexing with tensor mask. Note that the filler elements do not need to be set to False in the mask
    # as the sample sizes are assumed to be the same as the input data.
    input_mask_tensor = input_mask_ragged.tensor
    result = batched_bool_indexing(input_ragged, input_mask_tensor)
    assert isinstance(result, RaggedBatch)
    assert torch.all(result.sample_sizes == expected_sample_sizes)
    assert torch.equal(result.tensor, expected_output)

    # ----- Test with tensor input, RaggedBatch mask -----
    # Use the same data but swap input and mask. The sample sizes are assumed to be the same as the input mask.
    input_tensor = input_ragged.tensor
    result = batched_bool_indexing(input_tensor, input_mask_ragged)
    assert isinstance(result, RaggedBatch)
    assert torch.all(result.sample_sizes == expected_sample_sizes)
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_tensor_input(dtype):
    """Test batched_boolean_indexing with tensor input (and optionally RaggedBatch mask, both with 1 batch dim)."""
    # Use data from simple test
    input_ragged, input_mask_ragged, expected_output, expected_sample_sizes = (
        _create_simple_ragged_batch_test_data(dtype=dtype)
    )

    # ----- Test with tensor mask, tensor input -----
    # Convert both to tensors, setting filler values to False for mask. Note that as both inputs are tensors,
    # it is assumed that there are no filler values in the input data. Therefore, we need to make sure
    # that there are no `True` values in the mask in the filler positions, which is done using the `with_padded_set_to` method.
    input_tensor = input_ragged.tensor
    input_mask_tensor = input_mask_ragged.with_padded_set_to(False).tensor

    result = batched_bool_indexing(input_tensor, input_mask_tensor)
    assert isinstance(result, RaggedBatch)
    assert torch.all(result.sample_sizes == expected_sample_sizes)
    assert torch.equal(result.tensor, expected_output)

    # ----- Test with RaggedBatch mask, tensor input -----
    # Use the same data but keep mask as RaggedBatch
    result = batched_bool_indexing(input_tensor, input_mask_ragged)
    assert isinstance(result, RaggedBatch)
    assert torch.all(result.sample_sizes == expected_sample_sizes)
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_empty_selections(dtype):
    """Test batched_bool_indexing with empty selections (no elements selected)."""
    batch_size = 3
    max_sample_size = 4
    additional_dims = (2,)

    # Create input data
    input_data = torch.arange(
        batch_size * max_sample_size * np.prod(additional_dims), dtype=dtype, device="cuda:0"
    )
    input_data = input_data.reshape(batch_size, max_sample_size, *additional_dims)

    # Create sample sizes
    sample_sizes = torch.tensor([2, 3, 1], dtype=torch.int64, device="cuda:0")
    input_ragged = RaggedBatch(input_data, sample_sizes=sample_sizes)

    # Create mask with no selections
    input_mask = torch.zeros((batch_size, max_sample_size), dtype=torch.bool, device="cuda:0")

    # Apply boolean indexing
    result = batched_bool_indexing(input_ragged, input_mask)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check that all sample sizes are 0
    assert torch.all(result.sample_sizes == 0)

    # Check that the output tensor has the correct shape
    assert result.tensor.shape[0] == batch_size
    assert result.tensor.shape[1] == 0  # max sample size should be 0
    assert result.tensor.shape[2:] == additional_dims


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_all_selections(dtype):
    """Test batched_bool_indexing with all elements selected."""
    batch_size = 2
    max_sample_size = 3
    additional_dims = (2,)

    # Create input data
    input_data = torch.arange(
        batch_size * max_sample_size * np.prod(additional_dims), dtype=dtype, device="cuda:0"
    )
    input_data = input_data.reshape(batch_size, max_sample_size, *additional_dims)

    # Create sample sizes
    sample_sizes = torch.tensor([2, 3], dtype=torch.int64, device="cuda:0")
    input_ragged = RaggedBatch(input_data, sample_sizes=sample_sizes)

    # Create mask selecting all valid elements
    input_mask = torch.zeros((batch_size, max_sample_size), dtype=torch.bool, device="cuda:0")
    input_mask[0, :2] = True  # Select all 2 valid elements from sample 0
    input_mask[1, :3] = True  # Select all 3 valid elements from sample 1

    # Apply boolean indexing
    result = batched_bool_indexing(input_ragged, input_mask)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check sample sizes
    expected_sample_sizes = torch.tensor([2, 3], dtype=torch.int64, device="cuda:0")
    assert torch.all(result.sample_sizes == expected_sample_sizes)

    # Create expected output
    max_output_size = 3
    expected_output = torch.full(
        (batch_size, max_output_size, *additional_dims), fill_value=0.0, dtype=dtype, device="cuda:0"
    )
    expected_output[0, 0] = input_data[0, 0]
    expected_output[0, 1] = input_data[0, 1]
    expected_output[1, 0] = input_data[1, 0]
    expected_output[1, 1] = input_data[1, 1]
    expected_output[1, 2] = input_data[1, 2]

    # Check that the output contains the selected values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_gradient_propagation(dtype):
    """Test that gradients are properly propagated through batched_bool_indexing."""
    input_ragged, input_mask_ragged, _, _ = _create_simple_ragged_batch_test_data(dtype=dtype)

    # Make input data require gradients
    input_ragged.requires_grad = True

    # Apply boolean indexing
    result = batched_bool_indexing(input_ragged, input_mask_ragged)

    # Check that result requires gradients
    assert result.requires_grad

    # Compute a loss such that the gradients differ between elements
    # Set filler values to 0 to prevent them from contributing to the loss & getting gradients
    result.set_padded_to(0.0)
    # Only valid values are non-zero
    selected_elements = result.tensor
    # More complex loss to ensure that gradients are not 1.0 everywhere
    loss = selected_elements.sum() + selected_elements.pow(2).sum()
    loss.backward()

    # Check that gradients are computed for input_data
    assert input_ragged.tensor.grad is not None
    assert input_ragged.tensor.grad.shape == input_ragged.tensor.shape

    # Check that gradients are not all the same (not just 1 everywhere)
    grad = input_ragged.tensor.grad

    # Function for computing the gradient reference
    def ref_grad_func(tensor):
        # Grad for function: tensor**2 + tensor
        grads_all = tensor * 2 + 1
        mask_dims_expanded_shape = input_mask_ragged.with_padded_set_to(False).tensor.shape + (1,) * (
            tensor.ndim - input_mask_ragged.tensor.ndim
        )
        mask_dims_expanded = input_mask_ragged.with_padded_set_to(False).tensor.reshape(
            mask_dims_expanded_shape
        )
        grads_selected = grads_all * mask_dims_expanded
        return grads_selected

    # Compute the reference gradient
    ref_grad = input_ragged.apply(ref_grad_func).with_padded_set_to(0.0).tensor
    # Check that the gradients are correct
    atol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert torch.allclose(grad, ref_grad, atol=atol, rtol=0), f"Gradients are not correct (dtype={dtype})"

    # Check that gradients are only non-zero for elements that were selected by the mask
    # Create a mask indicating which input elements were selected (only for the first two dimensions)
    selected_input_mask = torch.zeros_like(
        input_ragged.tensor[..., 0, 0], dtype=torch.bool
    )  # Shape: [batch_size, max_sample_size]
    for i in range(input_ragged.tensor.shape[0]):
        valid_size = input_ragged.sample_sizes[i]
        selected_input_mask[i, :valid_size] = input_mask_ragged.tensor[i, :valid_size]

    # Expand the mask to match the input data shape
    expanded_mask = selected_input_mask.unsqueeze(-1).unsqueeze(-1).expand_as(input_ragged.tensor)

    # Gradients should only be non-zero for selected elements
    non_selected_grads = grad[~expanded_mask]
    assert torch.allclose(
        non_selected_grads, torch.zeros_like(non_selected_grads), atol=atol
    ), f"Gradients should be zero for non-selected elements (dtype={dtype})"

    # Gradients should be non-zero for at least some selected elements
    selected_grads = grad[expanded_mask]
    assert torch.any(selected_grads != 0), "Gradients should be non-zero for selected elements"


def test_dtype_consistency():
    """Test that batched_bool_indexing works correctly with different data types."""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        input_ragged, input_mask_ragged, expected_output, expected_sample_sizes = (
            _create_simple_ragged_batch_test_data()
        )

        # Convert data to the specified dtype
        input_ragged = input_ragged.to(dtype)
        expected_output = expected_output.to(dtype)

        # Apply boolean indexing
        result = batched_bool_indexing(input_ragged, input_mask_ragged)

        # Check that result has the correct dtype
        assert result.dtype == dtype

        # Check sample sizes
        assert torch.all(result.sample_sizes == expected_sample_sizes)

        # Check output values
        assert torch.equal(result.tensor, expected_output)


# -------------------------------------------------------------------------------------------------
# Inverse indexing tests (batched_bool_indexing_write)
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_simple_inverse_indexing(dtype):
    """Test batched_bool_indexing_write with simple RaggedBatch input."""
    to_write, mask_for_output, to_write_into, expected_output = _create_simple_inverse_indexing_test_data(
        dtype=dtype
    )

    # Apply inverse boolean indexing
    result = batched_bool_indexing_write(to_write, mask_for_output, to_write_into)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check output tensor shape
    assert result.tensor.shape == expected_output.shape

    # Check output values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_complex_multi_dim_inverse_indexing(dtype):
    """Test batched_bool_indexing_write with complex multi-dimensional RaggedBatch input."""
    to_write, mask_for_output, to_write_into, expected_output = (
        _create_complex_multi_dim_inverse_indexing_test_data(dtype=dtype)
    )

    # Apply inverse boolean indexing
    result = batched_bool_indexing_write(to_write, mask_for_output, to_write_into)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check output tensor shape
    assert result.tensor.shape == expected_output.shape

    # Check output values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_complex_multi_dim_inverse_indexing_transposed(dtype):
    """Test batched_bool_indexing_write with complex multi-dimensional RaggedBatch input where the non-uniform dimension is transposed."""
    to_write, mask_for_output, to_write_into, expected_output = (
        _create_complex_multi_dim_inverse_indexing_test_data(dtype=dtype)
    )

    # Transpose the non-uniform dimension to the last dimension (position 4)
    # Original shape: (2, 3, 4, 2, 3) with non_uniform_dim=2
    # Transposed shape: (2, 3, 2, 3, 4) with non_uniform_dim=4
    to_write_into_transposed = to_write_into.get_non_uniform_dimension_transposed_to(4)

    # Apply inverse boolean indexing
    result = batched_bool_indexing_write(to_write, mask_for_output, to_write_into_transposed)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check that the non-uniform dimension is preserved in the result
    assert result.non_uniform_dim == 4

    # Transpose the result back to the original non_uniform_dim (position 2)
    # This ensures that the result has the same structure as the expected output
    result_transposed_back = result.get_non_uniform_dimension_transposed_to(2)

    # Check that the transposed result matches the expected output
    assert torch.equal(result_transposed_back.tensor, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_inverse_indexing_tensor_to_write_into(dtype):
    """Test batched_bool_indexing_write when to_write_into is a tensor."""
    to_write, mask_for_output, to_write_into, expected_output = _create_simple_inverse_indexing_test_data(
        dtype=dtype
    )

    # ----- Test with tensor to_write_into -----
    # Convert to_write_into to tensor
    to_write_into_tensor = to_write_into.tensor
    result = batched_bool_indexing_write(to_write, mask_for_output, to_write_into_tensor)

    # Check that result is a tensor
    assert isinstance(result, torch.Tensor)

    # Check output tensor shape
    assert result.shape == expected_output.shape

    # Check output values
    assert torch.equal(result, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_inverse_indexing_tensor_mask(dtype):
    """Test batched_bool_indexing_write when mask_for_output is a tensor."""
    to_write, mask_for_output, to_write_into, expected_output = _create_simple_inverse_indexing_test_data(
        dtype=dtype
    )

    # ----- Test with tensor mask_for_output -----
    # Note that we do not set the filler values to False, as the filler values should be
    # handled correctly due to to_write_into being a RaggedBatch
    mask_for_output_tensor = mask_for_output.tensor

    result = batched_bool_indexing_write(to_write, mask_for_output_tensor, to_write_into)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check output tensor shape
    assert result.tensor.shape == expected_output.shape

    # Check output values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_inverse_indexing_both_tensors(dtype):
    """Test batched_bool_indexing_write when both mask_for_output and to_write_into are tensors."""
    to_write, mask_for_output, to_write_into, expected_output = _create_simple_inverse_indexing_test_data(
        dtype=dtype
    )

    # ----- Test with both tensors -----
    # Convert both to tensors
    to_write_into_tensor = to_write_into.tensor
    # For tensor mask, we need to ensure no True values at filler positions. This is needed as
    # both to_write_into and mask_for_output are tensors (filler values are handled automatically
    # if at least one of the inputs is a RaggedBatch).
    mask_for_output_tensor = mask_for_output.with_padded_set_to(False).tensor

    result = batched_bool_indexing_write(to_write, mask_for_output_tensor, to_write_into_tensor)

    # Check that result is a tensor
    assert isinstance(result, torch.Tensor)

    # Check output tensor shape
    assert result.shape == expected_output.shape

    # Check output values
    assert torch.equal(result, expected_output)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_inverse_indexing_empty_selections(dtype):
    """Test batched_bool_indexing_write with empty selections (no elements to write)."""
    batch_size = 3
    max_sample_size = 4
    additional_dims = (2,)

    # Create to_write data (empty)
    to_write_data = torch.zeros((batch_size, 0, *additional_dims), dtype=dtype, device="cuda:0")
    to_write_sample_sizes = torch.tensor([0, 0, 0], dtype=torch.int64, device="cuda:0")
    to_write = RaggedBatch(to_write_data, sample_sizes=to_write_sample_sizes)

    # Create to_write_into data
    to_write_into_data = torch.arange(
        batch_size * max_sample_size * np.prod(additional_dims), dtype=dtype, device="cuda:0"
    )
    to_write_into_data = to_write_into_data.reshape(batch_size, max_sample_size, *additional_dims)
    to_write_into_sample_sizes = torch.tensor([2, 3, 1], dtype=torch.int64, device="cuda:0")
    to_write_into = RaggedBatch(to_write_into_data, sample_sizes=to_write_into_sample_sizes)

    # Create mask with no selections
    mask_for_output = torch.zeros((batch_size, max_sample_size), dtype=torch.bool, device="cuda:0")
    mask_for_output_ragged = RaggedBatch(mask_for_output, sample_sizes=to_write_into_sample_sizes)

    # Apply inverse boolean indexing
    result = batched_bool_indexing_write(to_write, mask_for_output_ragged, to_write_into)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check that the output is unchanged (no writing occurred)
    assert torch.equal(result.tensor, to_write_into.tensor)
    assert torch.all(result.sample_sizes == to_write_into.sample_sizes)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_inverse_indexing_all_selections(dtype):
    """Test batched_bool_indexing_write with all elements selected."""
    batch_size = 2
    max_sample_size = 3
    additional_dims = (2,)

    # Create to_write data
    to_write_data = torch.arange(
        batch_size * max_sample_size * np.prod(additional_dims), dtype=dtype, device="cuda:0"
    )
    to_write_data = to_write_data.reshape(batch_size, max_sample_size, *additional_dims)
    to_write_sample_sizes = torch.tensor([2, 3], dtype=torch.int64, device="cuda:0")
    to_write = RaggedBatch(to_write_data, sample_sizes=to_write_sample_sizes)

    # Create to_write_into data (different from to_write to see the effect)
    to_write_into_data = torch.full(
        (batch_size, max_sample_size, *additional_dims),
        fill_value=100.0,
        dtype=dtype,
        device="cuda:0",
    )
    to_write_into_sample_sizes = torch.tensor([2, 3], dtype=torch.int64, device="cuda:0")
    to_write_into = RaggedBatch(to_write_into_data, sample_sizes=to_write_into_sample_sizes)

    # Create mask selecting all valid elements
    mask_for_output = torch.zeros((batch_size, max_sample_size), dtype=torch.bool, device="cuda:0")
    mask_for_output[0, :2] = True  # Select all 2 valid elements from sample 0
    mask_for_output[1, :3] = True  # Select all 3 valid elements from sample 1
    mask_for_output_ragged = RaggedBatch(mask_for_output, sample_sizes=to_write_into_sample_sizes)

    # Apply inverse boolean indexing
    result = batched_bool_indexing_write(to_write, mask_for_output_ragged, to_write_into)

    # Check that result is a RaggedBatch
    assert isinstance(result, RaggedBatch)

    # Check sample sizes
    assert torch.all(result.sample_sizes == to_write_into_sample_sizes)

    # Create expected output
    expected_output = to_write_into.tensor.clone()
    expected_output[0, :2] = to_write_data[0, :2]  # Write first 2 elements from to_write
    expected_output[1, :3] = to_write_data[1, :3]  # Write first 3 elements from to_write

    # Check that the output contains the written values
    assert torch.equal(result.tensor, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_inverse_indexing_gradient_propagation(dtype):
    """Test that gradients are properly propagated through batched_bool_indexing_write."""
    to_write, mask_for_output, to_write_into, _ = _create_simple_inverse_indexing_test_data(dtype=dtype)

    # Make both to_write and to_write_into require gradients
    to_write.requires_grad = True
    to_write_into.requires_grad = True

    # Apply inverse boolean indexing
    result = batched_bool_indexing_write(to_write, mask_for_output, to_write_into)

    # Check that result requires gradients
    assert result.requires_grad

    # Compute a loss such that the gradients differ between elements
    # Set filler values to 0 to prevent them from contributing to the loss & getting gradients
    result.set_padded_to(0.0)
    # Only valid values are non-zero
    selected_elements = result.tensor
    # More complex loss to ensure that gradients are not 1.0 everywhere
    loss = selected_elements.sum() + selected_elements.pow(2).sum()
    loss.backward()

    # Check that gradients are computed for both inputs
    assert to_write.tensor.grad is not None
    assert to_write.tensor.grad.shape == to_write.tensor.shape
    assert to_write_into.tensor.grad is not None
    assert to_write_into.tensor.grad.shape == to_write_into.tensor.shape

    # Test gradients for to_write
    to_write_grad = to_write.tensor.grad

    # Expected gradients for to_write: 2 * orig_val + 1 for valid elements; 0 for filler elements
    expected_to_write_grad = to_write.apply(lambda x: 2 * x + 1).with_padded_set_to(0.0).tensor

    # Compare actual vs expected gradients for to_write
    atol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert torch.allclose(
        to_write_grad, expected_to_write_grad, atol=atol, rtol=0
    ), f"Gradients for to_write do not match expected values (dtype={dtype})"

    # Test gradients for to_write_into
    to_write_into_grad = to_write_into.tensor.grad

    # Expand mask_for_output to match the to_write_into data shape
    # IMPORTANT: Note that we set the filler values to `True`. This is because we will compute the
    # negation of the mask, and we want the filler values to be `False` there.
    output_mask_tensor_filler_true = mask_for_output.with_padded_set_to(True).tensor
    output_mask_expanded_filler_true = (
        output_mask_tensor_filler_true.unsqueeze(-1).unsqueeze(-1).expand_as(to_write_into.tensor)
    )

    # Manually compute expected gradients for to_write_into:
    # - 2 * orig_val + 1 for valid elements that are NOT written to (mask is False)
    # - 0 for positions where values are written to (mask is True)
    # - 0 for filler elements
    expected_to_write_into_grad = torch.zeros_like(to_write_into.tensor)
    # Positions that are valid and not written to
    not_written_positions = ~output_mask_expanded_filler_true
    expected_to_write_into_grad[not_written_positions] = 2 * to_write_into.tensor[not_written_positions] + 1

    # Compare actual vs expected gradients for to_write_into
    assert torch.allclose(
        to_write_into_grad, expected_to_write_into_grad, atol=atol, rtol=0
    ), f"Gradients for to_write_into do not match expected values (dtype={dtype})"


def test_inverse_indexing_dtype_consistency():
    """Test that batched_bool_indexing_write works correctly with different data types."""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        to_write, mask_for_output, to_write_into, expected_output = (
            _create_simple_inverse_indexing_test_data()
        )

        # Convert data to the specified dtype
        to_write = to_write.to(dtype)
        to_write_into = to_write_into.to(dtype)
        expected_output = expected_output.to(dtype)

        # Apply inverse boolean indexing
        result = batched_bool_indexing_write(to_write, mask_for_output, to_write_into)

        # Check that result has the correct dtype
        assert result.dtype == dtype

        # Check output values
        assert torch.equal(result.tensor, expected_output)


if __name__ == "__main__":
    pytest.main([__file__])
