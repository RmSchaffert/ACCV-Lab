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
from accvlab.batching_helpers.batched_processing_py import RaggedBatch
from accvlab.batching_helpers.batched_mask_from_indices import get_mask_from_indices

# -------------------------------------------------------------------------------------------------
# Manual test data generation
# -------------------------------------------------------------------------------------------------


def _create_test_data():
    indices = torch.tensor(
        [
            [2, 0, 3, -1],
            [1, 2, -1, -1],
            [0, 3, 2, 4],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    # Set the invalid entries to a large value. This value should not be used for any purpose,
    # but setting it to a large value ensures that errors will be prodiced if it is used as an
    # index.
    indices[indices == -1] = 100

    nums_indices = torch.tensor([3, 2, 4], dtype=torch.int64, device="cuda:0")
    indices_batch = RaggedBatch(indices, sample_sizes=nums_indices)

    expected_mask = torch.tensor(
        [
            [True, False, True, True, False],
            [False, True, True, False, False],
            [True, False, True, True, True],
        ],
        dtype=torch.bool,
        device="cuda:0",
    )

    return indices_batch, expected_mask


def _create_test_data_multi_batch_dim():
    indices = torch.tensor(
        [
            [
                [2, 0, 3, -1],
                [1, 2, -1, -1],
                [0, 3, 2, 4],
            ],
            [
                [3, -1, -1, -1],
                [0, 3, 1, -1],
                [-1, -1, -1, -1],
            ],
        ],
        dtype=torch.int64,
        device="cuda:0",
    )
    # Set the invalid entries to a large value. This value should not be used for any purpose,
    # but setting it to a large value ensures that errors will be prodiced if it is used as an
    # index.
    indices[indices == -1] = 100

    nums_indices = torch.tensor([[3, 2, 4], [1, 3, 0]], dtype=torch.int64, device="cuda:0")
    indices_batch = RaggedBatch(indices, sample_sizes=nums_indices)

    expected_mask = torch.tensor(
        [
            [
                [True, False, True, True, False],
                [False, True, True, False, False],
                [True, False, True, True, True],
            ],
            [
                [False, False, False, True, False],
                [True, True, False, True, False],
                [False, False, False, False, False],
            ],
        ],
        dtype=torch.bool,
        device="cuda:0",
    )

    return indices_batch, expected_mask


# -------------------------------------------------------------------------------------------------
# Reference implementations for testing using random data
# -------------------------------------------------------------------------------------------------


def _reference_get_mask_from_indices(num_targets: int, indices: RaggedBatch) -> torch.Tensor:
    assert indices.dim() == 2, "The indices must be 2D"
    batch_size = indices.shape[0]
    mask = torch.zeros((batch_size, num_targets), dtype=torch.bool, device=indices.device)
    indices_data = indices.tensor.to(dtype=torch.int64)
    for i in range(batch_size):
        num_targets_curr = indices.sample_sizes[i].item()
        if num_targets_curr > 0:
            mask[i][indices_data[i, 0:num_targets_curr]] = True
    return mask


# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------


def test_batched_mask_from_indices_manual_example(capsys):
    indices_batch, expected_mask = _create_test_data()
    mask = get_mask_from_indices(5, indices_batch)
    assert torch.all(mask == expected_mask), "The mask is not correct"


def test_batched_mask_from_indices_multi_batch_dim_manual_example(capsys):
    indices_batch, expected_mask = _create_test_data_multi_batch_dim()
    mask = get_mask_from_indices(5, indices_batch)
    assert torch.all(mask == expected_mask), "The mask is not correct"


def test_batched_mask_from_indices_random_runs(capsys):
    # To see console outputs, use `with capsys.disabled(): ...`
    batch_size = 12
    num_tries = 1000
    num_indices = 10
    mask_size = 50
    for _ in range(num_tries):
        indices = torch.zeros((batch_size, num_indices), dtype=torch.int64, device="cuda:0")
        nums_indices = torch.randint(0, num_indices + 1, (batch_size,), dtype=torch.int64, device="cuda:0")

        # Ensure that the maximum number of indices corresponds to the size of `indices`
        if nums_indices.max() < num_indices:
            to_extend_idx = torch.randint(0, batch_size, (1,)).item()
            nums_indices[to_extend_idx] = num_indices

        # Fill the indices with random values
        for s in range(batch_size):
            use_negative = torch.randint(0, 2, (1,)).item() > 0
            ne = nums_indices[s].item()
            if use_negative:
                index_range = range(-mask_size, 0)
            else:
                index_range = range(0, mask_size)
            sampled_indices = np.random.choice(index_range, ne, replace=False)
            indices[s, 0:ne] = torch.tensor(sampled_indices)

        indices_batch = RaggedBatch(indices, sample_sizes=nums_indices)
        mask = get_mask_from_indices(mask_size, indices_batch)
        ref = _reference_get_mask_from_indices(mask_size, indices_batch)
        assert torch.all(mask == ref), "The mask is not correct"


if __name__ == "__main__":
    pytest.main([__file__])
