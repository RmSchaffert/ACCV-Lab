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

# -------------------------------------------------------------------------------------------------
# Test data generation
# -------------------------------------------------------------------------------------------------


def _create_test_data(dtype: torch.dtype = torch.float32):
    value_to_set = -10.0

    data = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        dtype=dtype,
        device="cuda:0",
    )
    sample_sizes = torch.tensor([3, 2, 5], dtype=torch.int64, device="cuda:0")
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    expected_data = torch.tensor(
        [
            [1.0, 2.0, 3.0, -10.0, -10.0],
            [6.0, 7.0, -10.0, -10.0, -10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    if torch.is_floating_point(torch.empty((), dtype=dtype)):
        nan = torch.nan
        expected_grad = torch.cos(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, nan, nan],
                    [6.0, 7.0, nan, nan, nan],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_grad = torch.nan_to_num(expected_grad, nan=0.0)
    else:
        expected_grad = None

    return ragged_batch, value_to_set, expected_data, expected_grad


def _create_test_data_multi_batch_dim(dtype: torch.dtype = torch.float32):
    value_to_set = -10.0

    data = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            [
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [26.0, 27.0, 28.0, 29.0, 30.0],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )
    sample_sizes = torch.tensor([[3, 2, 5], [4, 0, 1]], dtype=torch.int64, device="cuda:0")
    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    expected_data = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, -10.0, -10.0],
                [6.0, 7.0, -10.0, -10.0, -10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            [
                [16.0, 17.0, 18.0, 19.0, -10.0],
                [-10.0, -10.0, -10.0, -10.0, -10.0],
                [26.0, -10.0, -10.0, -10.0, -10.0],
            ],
        ],
        dtype=dtype,
        device="cuda:0",
    )

    if torch.is_floating_point(torch.empty((), dtype=dtype)):
        nan = torch.nan
        expected_grad = torch.cos(
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 3.0, nan, nan],
                        [6.0, 7.0, nan, nan, nan],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                    ],
                    [
                        [16.0, 17.0, 18.0, 19.0, nan],
                        [nan, nan, nan, nan, nan],
                        [26.0, nan, nan, nan, nan],
                    ],
                ],
                dtype=dtype,
                device="cuda:0",
            )
        )
        expected_grad = torch.nan_to_num(expected_grad, nan=0.0)
    else:
        expected_grad = None

    return ragged_batch, value_to_set, expected_data, expected_grad


def _create_random_test_data(
    batch_size: int,
    max_sample_size: int,
    additional_shape: tuple,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
):
    """Creates random test data for ragged batch set padded to tests.

    Args:
        batch_size: Number of samples in the batch
        max_sample_size: Maximum number of elements per sample
        additional_shape: Additional dimensions for the data tensor
        requires_grad: Whether to enable gradients on the data

    Returns:
        tuple: (data, sample_sizes, ragged_batch, ragged_batch_ref, ragged_batch_res)
    """
    if torch.is_floating_point(torch.empty((), dtype=dtype)):
        data = torch.randn((batch_size, max_sample_size, *additional_shape), device="cuda:0", dtype=dtype)
    else:
        data = torch.randint(
            low=-1000,
            high=1000,
            size=(batch_size, max_sample_size, *additional_shape),
            device="cuda:0",
            dtype=dtype,
        )
    sample_sizes = torch.randint(0, max_sample_size + 1, (batch_size,), device="cuda:0")
    # Ensure that the maximum number of elements corresponds to the size of the `data`
    if sample_sizes.max().item() < max_sample_size:
        to_extend_idx = torch.randint(0, batch_size, (1,)).item()
        sample_sizes[to_extend_idx] = max_sample_size

    ragged_batch = RaggedBatch(data, sample_sizes=sample_sizes)

    return data, sample_sizes, ragged_batch


# -------------------------------------------------------------------------------------------------
# Reference implementations for testing using random data
# -------------------------------------------------------------------------------------------------


def _reference_set_padded_to(input: RaggedBatch, value_to_set: float) -> RaggedBatch:
    neg_mask = torch.logical_not(input.mask)
    data = input.tensor
    if input._non_uniform_dim != 1:
        data = data.transpose(1, input._non_uniform_dim)
    data[neg_mask] = value_to_set
    if input._non_uniform_dim != 1:
        data = data.transpose(1, input._non_uniform_dim)
    input.set_tensor(data)


# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_ragged_batch_set_padded_to_forward_examples(dtype, capsys):
    ragged_batch, value_to_set, expected_data, _ = _create_test_data(dtype=dtype)

    # Compute the result using the implementation
    ragged_batch.set_padded_to(value_to_set)

    # Verify the result
    diff = ragged_batch.tensor - expected_data
    max_abs_diff = torch.max(torch.abs(diff))

    # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
    # errors and the results should match exactly.
    assert (
        max_abs_diff == 0.0
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_ragged_batch_set_padded_to_forward_examples_multi_batch_dim(dtype, capsys):
    ragged_batch, value_to_set, expected_data, _ = _create_test_data_multi_batch_dim(dtype=dtype)

    # Compute the result using the implementation
    ragged_batch.set_padded_to(value_to_set)

    # Verify the result
    diff = ragged_batch.tensor - expected_data
    max_abs_diff = torch.max(torch.abs(diff))

    # Here, no actual computations are performed but individual values are only copied. This means that there are no numerical
    # errors and the results should match exactly.
    assert (
        max_abs_diff == 0.0
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_ragged_batch_set_padded_to_backward_example(dtype, capsys):
    ragged_batch, value_to_set, _, expected_grad = _create_test_data(dtype=dtype)

    # Enable gradients
    ragged_batch.requires_grad = True
    ragged_batch.retain_grad()

    # Compute the result using the implementation.
    # Note that a cloned version of the data is used, as the data is a leaf variable and and
    # we need it later to get the gradients.
    ragged_batch_to_apply = ragged_batch.as_self_with_cloned_data()
    ragged_batch_to_apply.set_padded_to(value_to_set)

    # Make sure the gradients are not identical for all elements
    res_forward_proc = torch.sin(ragged_batch_to_apply.tensor)
    res_sum = torch.sum(res_forward_proc)
    res_sum.backward()

    # Verify the gradients
    grad_diff = ragged_batch.tensor.grad - expected_grad
    max_abs_diff = torch.max(torch.abs(grad_diff))

    tol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert (
        max_abs_diff < tol
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_ragged_batch_set_padded_to_backward_example_multi_batch_dim(dtype, capsys):
    ragged_batch, value_to_set, _, expected_grad = _create_test_data_multi_batch_dim(dtype=dtype)

    # Enable gradients
    ragged_batch.requires_grad = True
    ragged_batch.retain_grad()

    # Compute the result using the implementation.
    # Note that a cloned version of the data is used, as the data is a leaf variable and and
    # we need it later to get the gradients.
    ragged_batch_to_apply = ragged_batch.as_self_with_cloned_data()
    ragged_batch_to_apply.set_padded_to(value_to_set)

    # Make sure the gradients are not identical for all elements
    res_forward_proc = torch.sin(ragged_batch_to_apply.tensor)
    res_sum = torch.sum(res_forward_proc)
    res_sum.backward()

    # Verify the gradients
    grad_diff = ragged_batch.tensor.grad - expected_grad
    max_abs_diff = torch.max(torch.abs(grad_diff))

    tol = 1e-6 if dtype in (torch.float32, torch.float64) else 1e-3
    assert (
        max_abs_diff < tol
    ), f"Difference {max_abs_diff} between implementation and reference detected (dtype={dtype})"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_ragged_batch_set_padded_to_forward(dtype, capsys):
    # To see console outputs, use `with capsys.disabled(): ...`
    batch_size = 12
    max_sample_size = 50
    num_tries = 1000
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        _, _, ragged_batch = _create_random_test_data(
            batch_size, max_sample_size, additional_shape, dtype=dtype
        )
        ragged_batch_ref = ragged_batch.as_self_with_cloned_data()
        ragged_batch_res = ragged_batch

        value_to_set = torch.randn(1).item()
        ragged_batch_res.set_padded_to(value_to_set)
        _reference_set_padded_to(ragged_batch_ref, value_to_set)

        data_ref = ragged_batch_ref.tensor
        data_res = ragged_batch_res.tensor
        diff = torch.abs(data_ref - data_res)
        max_diff = torch.max(diff)
        # Note that here, we can check for equality, as there are no floating point operations involved,
        # but only assignments to the filler values.
        assert max_diff == 0.0, f"Differences in padded values detected. Maximum difference: {max_diff}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_ragged_batch_set_padded_to_backward(dtype, capsys):
    # To see console outputs, use `with capsys.disabled(): ...`
    batch_size = 12
    max_sample_size = 50
    num_tries = 1000
    additional_shape = (2, 50, 3)

    for _ in range(num_tries):
        _, _, ragged_batch = _create_random_test_data(
            batch_size, max_sample_size, additional_shape, requires_grad=True, dtype=dtype
        )

        # Note that the `_in` indicates that this is the input data, i.e. before
        # the operation is applied.
        ragged_batch.requires_grad = True
        ragged_batch_res_in = ragged_batch
        with torch.no_grad():
            ragged_batch_ref_in = ragged_batch.as_self_with_cloned_data()
        ragged_batch_ref_in.requires_grad = True
        ragged_batch_ref_in.retain_grad()
        ragged_batch_res_in.retain_grad()

        value_to_set = torch.randn(1).item()

        # Note that cloned versions of the data is used, as the original versions are leaf variables and
        # we need them later to get the gradients.
        ragged_batch_res = ragged_batch_res_in.as_self_with_cloned_data()
        ragged_batch_res.set_padded_to(value_to_set)
        ragged_batch_ref = ragged_batch_ref_in.as_self_with_cloned_data()
        _reference_set_padded_to(ragged_batch_ref, value_to_set)

        # Make sure that the gradients are not identical for all elements
        ragged_batch_res.apply(lambda data: torch.sin(data))
        ragged_batch_ref.apply(lambda data: torch.sin(data))

        # Compute the "loss" and start the backward pass.
        loss = torch.sum(ragged_batch_res.tensor)
        loss.backward()
        loss_ref = torch.sum(ragged_batch_ref.tensor)
        loss_ref.backward()

        grad_data_res = ragged_batch_res_in.tensor.grad
        grad_data_ref = ragged_batch_ref_in.tensor.grad

        diff_grad_data = torch.abs(grad_data_res - grad_data_ref)
        max_diff_grad_data = torch.max(diff_grad_data)
        # Note that here, we can check for equality, as there are no floating point operations involved,
        # but only assignments to the filler values.
        assert (
            max_diff_grad_data == 0.0
        ), f"Differences in gradients detected. Maximum difference: {max_diff_grad_data}"


if __name__ == "__main__":
    pytest.main([__file__])
