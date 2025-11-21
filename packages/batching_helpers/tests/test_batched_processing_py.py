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
from accvlab.batching_helpers.batched_processing_py import RaggedBatch
import accvlab.batching_helpers.batched_processing_py as bbp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_average_over_targets(dtype, capsys):
    """Test averaging over targets in ragged batch"""
    batch_shape = (2, 3)  # 2x3 batch grid
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device, dtype=dtype)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test averaging over targets
    averaged = bbp.average_over_targets(ragged_batch)

    # Check shapes and dtype
    assert averaged.shape == (*batch_shape, *additional_shape)
    assert averaged.dtype == dtype

    tol = 1e-6 if dtype in (torch.float32, torch.float64) else (1e-3 if dtype == torch.float16 else 1e-2)

    # Check that averaging was done correctly
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            expected_avg = torch.mean(data[i, j, : sample_sizes[i, j]], dim=0)
            assert torch.allclose(averaged[i, j], expected_avg, atol=tol, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_sum_over_targets(dtype, capsys):
    """Test summing over targets in ragged batch"""
    batch_shape = (2, 3)  # 2x3 batch grid
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device, dtype=dtype)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test summing over targets
    summed = bbp.sum_over_targets(ragged_batch)

    # Check shapes and dtype
    assert summed.shape == (*batch_shape, *additional_shape)
    assert summed.dtype == dtype

    tol = 1e-6 if dtype in (torch.float32, torch.float64) else (1e-3 if dtype == torch.float16 else 1e-2)

    # Check that summing was done correctly
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            expected_sum = torch.sum(data[i, j, : sample_sizes[i, j]], dim=0)
            assert torch.allclose(summed[i, j], expected_sum, atol=tol, rtol=0)


def test_apply_mask_to_tensor(capsys):
    """Test applying mask to tensor"""
    batch_shape = (2, 3)  # 2x3 batch grid
    max_sample_size = 5
    additional_shape = (2,)
    device = "cuda:0"  # Only test on GPU

    # Create test data and mask
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    mask = torch.zeros((*batch_shape, max_sample_size), dtype=torch.bool, device=device)

    # Add some NaN values to the region which corresponds to the mask being false
    # to ensure that these values are also set to the custom value
    data_with_nans = data.clone()
    data_with_nans[0, 0, 1, :] = torch.nan
    data_with_nans[0, 1, [2, 4], :] = torch.nan
    data_with_nans[1, 0, [3, 4], :] = torch.nan
    data_with_nans[1, 1, 0, :] = torch.nan

    # Set mask values to True in non-sequential pattern
    mask[0, 0, [0, 2, 4]] = True  # First sample: elements 0, 2, 4 valid
    mask[0, 1, [1, 3]] = True  # Second sample: elements 1, 3 valid
    mask[0, 2, [0, 1, 2, 3, 4]] = True  # Third sample: all elements valid
    mask[1, 0, [0, 1, 2]] = True  # Fourth sample: first 3 elements valid
    mask[1, 1, [2, 3, 4]] = True  # Fifth sample: last 3 elements valid
    mask[1, 2, [0, 2, 4]] = True  # Sixth sample: elements 0, 2, 4 valid

    # Test applying mask with default value (0.0)
    masked_data = bbp.apply_mask_to_tensor(data_with_nans, mask)

    # Check shapes
    assert masked_data.shape == data.shape

    # Check that mask was applied correctly
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            # Valid entries should be unchanged
            valid_indices = torch.where(mask[i, j])[0]
            assert torch.all(masked_data[i, j, valid_indices] == data[i, j, valid_indices])
            # Invalid entries should be zero
            invalid_indices = torch.where(~mask[i, j])[0]
            assert torch.all(masked_data[i, j, invalid_indices] == 0.0)

    # Test with custom value
    custom_value = -1.0
    masked_data = bbp.apply_mask_to_tensor(data, mask, value_to_set=custom_value)

    # Check that mask was applied correctly with custom value
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            # Valid entries should be unchanged
            valid_indices = torch.where(mask[i, j])[0]
            assert torch.all(masked_data[i, j, valid_indices] == data[i, j, valid_indices])
            # Invalid entries should be set to custom value
            invalid_indices = torch.where(~mask[i, j])[0]
            assert torch.all(masked_data[i, j, invalid_indices] == custom_value)

    # Test with mask having fewer dimensions
    reduced_mask = torch.zeros(batch_shape, dtype=torch.bool, device=device)
    reduced_mask[0, 0] = True  # Only first sample is valid

    masked_data = bbp.apply_mask_to_tensor(data, reduced_mask)

    # Check that reduced mask was applied correctly
    assert torch.all(masked_data[0, 0] == data[0, 0])  # First sample unchanged
    assert torch.all(masked_data[0, 1:] == 0.0)  # Other samples in first batch zeroed
    assert torch.all(masked_data[1:] == 0.0)  # All samples in second batch zeroed


def test_squeeze_except_batch_and_sample(capsys):
    """Test squeezing dimensions except batch and non-uniform dimensions"""
    batch_shape = (2, 3)  # Multiple batch dimensions
    max_sample_size = 5
    additional_shape = (1, 2, 1)  # Shape with dimensions that can be squeezed
    device = "cuda:0"  # Only test on GPU

    # Create ragged batch
    data = torch.randn((*batch_shape, max_sample_size, *additional_shape), device=device)
    sample_sizes = torch.tensor([[3, 2, 5], [1, 2, 3]], dtype=torch.int64, device=device)
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    # Test squeezing RaggedBatch
    squeezed_batch = bbp.squeeze_except_batch_and_sample(ragged_batch)

    # Check shapes - should remove dimensions of size 1
    expected_shape = (*batch_shape, max_sample_size, 2)  # Removed the 1s
    assert squeezed_batch.tensor.shape == expected_shape
    # Check that non-uniform dimension is preserved (nothing squeezed before this dimension)
    assert squeezed_batch.non_uniform_dim == 2  # Should be 2 after squeezing (after batch dims)

    # Check that data is preserved
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            assert torch.allclose(
                squeezed_batch.tensor[i, j, : sample_sizes[i, j]], data[i, j, : sample_sizes[i, j]].squeeze()
            )

    # Test with tensor input
    squeezed_tensor = bbp.squeeze_except_batch_and_sample(data)
    assert squeezed_tensor.shape == expected_shape

    # Test with tensor that has non-uniform dimension at different position
    ragged_batch_transposed = ragged_batch.get_non_uniform_dimension_transposed_to(3)
    squeezed_transposed = bbp.squeeze_except_batch_and_sample(ragged_batch_transposed)
    expected_shape_transposed = (*batch_shape, max_sample_size, 2)
    assert squeezed_transposed.shape == expected_shape_transposed

    # Test with batch size and sample sizes of size 1
    data_single = torch.randn((1, 1, 1, 1), device=device)
    sample_sizes_single = torch.tensor([[1]], dtype=torch.int64, device=device)
    ragged_batch_single = RaggedBatch(data_single, sample_sizes=sample_sizes_single)
    squeezed_single = bbp.squeeze_except_batch_and_sample(ragged_batch_single)
    assert squeezed_single.tensor.shape == (1, 1, 1)


def test_combine_data_with_flattening(capsys):
    """Test combining data into a single RaggedBatch"""
    device = "cuda:0"  # Only test on GPU

    # Test combining tensors with different sample sizes
    data1 = torch.randn((3,), device=device)  # Sample with 3 elements
    data2 = torch.randn((2,), device=device)  # Sample with 2 elements
    data3 = torch.randn((4,), device=device)  # Sample with 4 elements

    # Test combining with reference batch
    combined = bbp.combine_data([data1, data2, data3])
    assert combined.tensor.shape == (3, 4)  # 3 samples, max size 4

    # Check that data is preserved correctly for each sample
    assert torch.allclose(combined.tensor[0, :3], data1)
    assert torch.allclose(combined.tensor[1, :2], data2)
    assert torch.allclose(combined.tensor[2, :4], data3)

    # Test combining nested sequences with different sample sizes
    nested_data = [[data1, data2], [data3]]  # Nested structure
    combined_from_nested = bbp.combine_data(nested_data)
    assert combined_from_nested.tensor.shape == (3, 4)
    assert torch.allclose(combined_from_nested.tensor[0, :3], data1)
    assert torch.allclose(combined_from_nested.tensor[1, :2], data2)
    assert torch.allclose(combined_from_nested.tensor[2, :4], data3)

    # Test combining tensors with different shapes and sample sizes
    data4 = torch.randn((2, 4), device=device)  # Sample with 2 elements and extra dimension
    data5 = torch.randn((3, 4), device=device)  # Sample with 3 elements and extra dimension
    combined_different = bbp.combine_data([data4, data5])
    assert combined_different.tensor.shape == (2, 3, 4)  # 2 samples, max size 3, 4 elements
    assert torch.allclose(combined_different.tensor[0, :2, :], data4)
    assert torch.allclose(combined_different.tensor[1, :3, :], data5)

    # Test combining with empty sequence
    with pytest.raises(AssertionError):
        bbp.combine_data([])

    # Test combining with non-tensor elements
    mixed_data = [data1, "not a tensor", data2]
    with pytest.raises(AssertionError):
        bbp.combine_data(mixed_data)


def test_combine_data_with_batch_shape_preservation(capsys):
    """Test combining data while preserving batch shapes"""
    device = "cuda:0"  # Only test on GPU

    batch_shape = (2, 3)

    data = [[None] * batch_shape[1] for _ in range(batch_shape[0])]

    # Test combining tensors with different batch shapes
    data[0][0] = torch.randn((2, 3, 4), device=device)
    data[0][1] = torch.randn((1, 3, 4), device=device)
    data[0][2] = torch.randn((7, 3, 4), device=device)
    data[1][0] = torch.randn((5, 3, 4), device=device)
    data[1][1] = torch.randn((3, 3, 4), device=device)
    data[1][2] = torch.randn((0, 3, 4), device=device)

    # Test combining with same batch shape
    combined = bbp.combine_data(data, flatten_batch_dims=False)
    assert combined.batch_shape == batch_shape
    assert combined.shape == (*batch_shape, 7, 3, 4)
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            assert torch.allclose(combined.tensor[i, j, : data[i][j].shape[0]], data[i][j])
            assert combined.sample_sizes[i, j] == data[i][j].shape[0]

    # Test combining with different batch shapes
    data_additional = torch.randn((3, 3, 4), device=device)  # 3x2 batch, 4 elements per sample
    with pytest.raises(AssertionError):
        bbp.combine_data(
            [data[0], data[1] + [data_additional]], flatten_batch_dims=False
        )  # Should fail due to non-uniform batch shape


def test_get_compact_functions(capsys):
    """Test compactification functions for both lists and named tuples"""
    device = "cuda:0"

    # Create test data
    data1 = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        device=device,
    )
    data2 = torch.tensor(
        [
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [25, 26, 27, 28],
            [29, 30, 31, 32],
        ],
        device=device,
    )
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, False],
            [True, True, True, False],
            [False, False, False, False],
        ],
        device=device,
    )

    # Test get_compact_lists
    compacted = bbp.get_compact_lists(mask, [data1, data2])

    # Check that we got RaggedBatch instances
    assert isinstance(compacted[0], bbp.RaggedBatch)
    assert isinstance(compacted[1], bbp.RaggedBatch)

    # Check shapes and data
    assert compacted[0].tensor.shape == (4, 3)  # 4 samples, max 3 valid elements
    assert compacted[1].tensor.shape == (4, 3)

    # Check that data is preserved correctly
    assert torch.all(compacted[0].sample_sizes == torch.tensor([2, 1, 3, 0], device=device))
    assert torch.all(compacted[1].sample_sizes == torch.tensor([2, 1, 3, 0], device=device))
    assert torch.allclose(compacted[0].tensor[0][:2], torch.tensor([1, 3], device=device))
    assert torch.allclose(compacted[0].tensor[1][:1], torch.tensor([6], device=device))
    assert torch.allclose(compacted[0].tensor[2][:3], torch.tensor([9, 10, 11], device=device))
    assert torch.allclose(compacted[1].tensor[0][:2], torch.tensor([17, 19], device=device))
    assert torch.allclose(compacted[1].tensor[1][:1], torch.tensor([22], device=device))
    assert torch.allclose(compacted[1].tensor[2][:3], torch.tensor([25, 26, 27], device=device))
    # Test get_compact_lists with non-tensor elements
    mixed_data = [data1, "not a tensor", data2]
    compacted_mixed = bbp.get_compact_lists(mask, mixed_data)
    assert isinstance(compacted_mixed[0], bbp.RaggedBatch)
    assert compacted_mixed[1] == "not a tensor"
    assert isinstance(compacted_mixed[2], bbp.RaggedBatch)

    # Test get_compact_from_named_tuple
    from collections import namedtuple

    TestTuple = namedtuple('TestTuple', ['data1', 'data2', 'non_tensor'])
    test_tuple = TestTuple(data1=data1, data2=data2, non_tensor="test")

    # Test compactification
    compacted_tuple = bbp.get_compact_from_named_tuple(mask, test_tuple)

    # Check that we got a named tuple of the same type
    assert isinstance(compacted_tuple, TestTuple)

    # Check that tensors were converted to RaggedBatch
    assert isinstance(compacted_tuple.data1, bbp.RaggedBatch)
    assert isinstance(compacted_tuple.data2, bbp.RaggedBatch)
    assert compacted_tuple.non_tensor == "test"

    # Check shapes and data
    assert compacted_tuple.data1.tensor.shape == (4, 3)
    assert compacted_tuple.data2.tensor.shape == (4, 3)

    # Check that data is preserved correctly
    assert torch.all(compacted_tuple.data1.sample_sizes == torch.tensor([2, 1, 3, 0], device=device))
    assert torch.all(compacted_tuple.data2.sample_sizes == torch.tensor([2, 1, 3, 0], device=device))
    assert torch.allclose(compacted_tuple.data1.tensor[0][:2], torch.tensor([1, 3], device=device))
    assert torch.allclose(compacted_tuple.data1.tensor[1][:1], torch.tensor([6], device=device))
    assert torch.allclose(compacted_tuple.data1.tensor[2][:3], torch.tensor([9, 10, 11], device=device))
    assert torch.allclose(compacted_tuple.data2.tensor[0][:2], torch.tensor([17, 19], device=device))
    assert torch.allclose(compacted_tuple.data2.tensor[1][:1], torch.tensor([22], device=device))
    assert torch.allclose(compacted_tuple.data2.tensor[2][:3], torch.tensor([25, 26, 27], device=device))


def test_get_indices_from_mask_tensor_mask(capsys):
    device = "cuda:0"

    # 2D boolean mask with varied patterns per sample
    mask = torch.tensor(
        [
            [True, False, True, False, False, True],  # -> [0, 2, 5]
            [False, True, False, False, False, False],  # -> [1]
            [False, False, False, False, False, False],  # -> []
            [True, True, True, True, True, True],  # -> [0,1,2,3,4,5]
        ],
        dtype=torch.bool,
        device=device,
    )

    indices = bbp.get_indices_from_mask(mask)

    # Check type and shapes
    assert isinstance(indices, RaggedBatch)
    assert indices.tensor.dtype == torch.int64
    assert indices.tensor.shape == (4, 6)

    # Check sample sizes
    expected_sizes = torch.tensor([3, 1, 0, 6], device=device)
    assert torch.equal(indices.sample_sizes, expected_sizes)

    # Check indexed values per row
    expected_rows = [
        torch.tensor([0, 2, 5], device=device),
        torch.tensor([1], device=device),
        torch.tensor([], dtype=torch.int64, device=device),
        torch.tensor([0, 1, 2, 3, 4, 5], device=device),
    ]
    for i, expected in enumerate(expected_rows):
        if expected.numel() == 0:
            assert indices.sample_sizes[i].item() == 0
        else:
            num = indices.sample_sizes[i].item()
            assert torch.equal(indices.tensor[i, :num], expected)


def test_get_indices_from_mask_ragged_mask(capsys):
    device = "cuda:0"

    # Build a RaggedBatch whose tensor holds the actual boolean mask.
    # Also set per-sample sizes so that some trailing True values lie beyond the
    # sample size and must be ignored as padded fillers.
    mask_tensor = torch.tensor(
        [
            # sample 0: only first two are valid; trailing True must be ignored by sample_sizes
            [True, True, True, False, True],
            # sample 1: all valid
            [True, False, True, True, True],
            # sample 2: no valid within sample size
            [True, True, False, True, False],
        ],
        dtype=torch.bool,
        device=device,
    )
    sample_sizes = torch.tensor([2, 5, 0], dtype=torch.int64, device=device)
    rb_mask = RaggedBatch(mask_tensor, sample_sizes=sample_sizes, non_uniform_dim=1)

    indices = bbp.get_indices_from_mask(rb_mask)

    # Expected sizes: only True positions within sample_sizes count
    expected_sizes = torch.tensor([2, 4, 0], dtype=torch.int64, device=device)

    # Validate basics
    assert isinstance(indices, RaggedBatch)
    assert indices.tensor.dtype == torch.int64
    # Width equals max number of selected indices, not mask width
    assert indices.tensor.shape == (3, int(expected_sizes.max().item()))

    # Only positions < sample_sizes that are True should be selected
    assert torch.equal(indices.sample_sizes, expected_sizes)

    expected_rows = [
        torch.tensor([0, 1], dtype=torch.int64, device=device),  # True at positions 0,1 within size 2
        torch.tensor(
            [0, 2, 3, 4], dtype=torch.int64, device=device
        ),  # positions 0,2,3 within size 4; pos 4 ignored
        torch.tensor([], dtype=torch.int64, device=device),  # size 0 -> empty
    ]
    for i, expected in enumerate(expected_rows):
        num = indices.sample_sizes[i].item()
        assert num == expected.numel()
        if num > 0:
            assert torch.equal(indices.tensor[i, :num], expected)


def test_get_indices_from_mask_invalid_dimensions(capsys):
    device = "cuda:0"
    # 3D mask should raise assertion
    bad_mask = torch.zeros((2, 3, 4), dtype=torch.bool, device=device)
    with pytest.raises(AssertionError):
        _ = bbp.get_indices_from_mask(bad_mask)


if __name__ == "__main__":
    pytest.main([__file__])
