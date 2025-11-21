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

import torch
from torch.autograd.function import once_differentiable
from typing import Any, Union
import accvlab.batching_helpers.batched_indexing_access_cuda as batched_indexing_access_cuda
from .data_format import RaggedBatch


class BatchedIndexMapping(torch.autograd.Function):
    """Implementation of mapping values between two tensors using a mapping defined by pairs of indices as well as the corresponding backward operation.

    Please see documentation of :func:`batched_index_mapping` for more details.

    Warning:
        This class is not intended to be used directly. It is an implementation detail of :func:`batched_index_mapping`.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_data: torch.Tensor,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        nums_indices: torch.Tensor,
        to_insert_into: torch.Tensor,
    ) -> torch.Tensor:
        input_data = input_data.contiguous()
        input_indices = input_indices.contiguous()
        output_indices = output_indices.contiguous()
        nums_indices = nums_indices.contiguous()
        to_insert_into = to_insert_into.contiguous()
        result = batched_indexing_access_cuda.map_values_by_index_pairs(
            input_data, input_indices, output_indices, nums_indices, to_insert_into, backward_accumulate=False
        )
        ctx.save_for_backward(input_indices, output_indices, nums_indices)
        ctx.num_batch_dims = nums_indices.dim()
        ctx.input_max_sample_size = input_data.shape[ctx.num_batch_dims]
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: Union[torch.Tensor, None]):
        if grad is None:
            return None, None, None, None, None
        else:
            forward_input_indices, forward_output_indices, nums_indices = ctx.saved_tensors
            # The shape for `grad_input` corresponds to the shape of `grad` except for `dim==1`,
            # where the size may be different. The size in this dim is known from the forward step.
            shape_to_set = list(grad.shape)
            shape_to_set[ctx.num_batch_dims] = ctx.input_max_sample_size

            grad = grad.contiguous()

            grad_input = torch.zeros(
                shape_to_set, dtype=grad.dtype, device=grad.device, requires_grad=grad.requires_grad
            )
            grad_input = batched_indexing_access_cuda.map_values_by_index_pairs(
                grad,
                forward_output_indices,
                forward_input_indices,
                nums_indices,
                grad_input,
                backward_accumulate=True,
            )
            grad_to_insert_into = batched_indexing_access_cuda.backward_insert_const(
                0.0, forward_output_indices, nums_indices, grad
            )
            return grad_input, None, None, None, grad_to_insert_into


def batched_index_mapping(
    source_data: Union[torch.Tensor, RaggedBatch],
    source_indices: RaggedBatch,
    target_indices: RaggedBatch,
    target_data: Union[torch.Tensor, RaggedBatch],
) -> Union[torch.Tensor, RaggedBatch]:
    """Map values between ``source_data`` and ``target_data`` using a mapping defined by pairs of indices for elements
    in ``source_data`` and ``target_data``, and set the corresponding values in ``target_data``.

    :gpu:

    For a sample ``i`` and a valid index pair ``j`` (valid means ``target_indices.sample_sizes[i] > j`` and
    ``source_indices.sample_sizes[i] > j``), the operation can be expressed as (assuming non-uniform dimension
    ``dim == 1`` for both ``source_data`` and ``target_data``):

        ``target_data[i, target_indices[i, j]] = source_data[i, source_indices[i, j]]``

    This function sets the values in `target_data` in a way which corresponds to the line above
    for all valid matches in all samples.

    Warning:

        It is expected that for each sample, the number of valid indices for the source and target matches,
        i.e. ``target_indices.sample_sizes == source_indices.sample_sizes``.

        If this is not the case, the behavior is undefined.

    Warning:

        This function assumes that for each sample, there are no duplicate indices in
        ``target_indices``, i.e. there are no duplicates in the valid entries in:
        ``target_indices[i, 0:target_indices.sample_sizes[i]]``.

        If this is not the case, the behavior is undefined.

        There are no such restrictions on `source_indices`.

    Args:
        source_data: Input data.

            Shape in case of a tensor:
                (batch_size, num_entries_input, ...)

            In case of a RaggedBatch, the following holds:

                - ``target_data.shape[0] == batch_size``
                - ``target_data.shape[target_data.non_uniform_dim] == num_entries_input``

            The number of dimensions needs to correspond to ``target_data``, and the shape needs
            to be the same except in the non-uniform dimension (``dim == 1`` for tensors), as in
            this dimension, the matching is done using the index pairs.

        source_indices: Indices at which to get the input data. Shape: (batch_size, max_num_matches).
            Note that the batch size and sample sizes need to match ``target_indices``, i.e.
            ``target_indices.sample_sizes == source_indices.sample_sizes`` and ``max_num_matches``
            corresponds to the maximum matches among all samples.

        target_indices: Indices at which to fill the data. Shape: (batch_size, max_num_matches)
            Note that ``max_num_matches`` corresponds to the maximum matches among all samples.

        target_data: Data to fill the values into.

            Shape in case of a tensor: (batch_size, num_entries_output, ...)

            In case of a RaggedBatch, the following holds:
                - ``target_data.shape[0] == batch_size``
                - ``target_data.shape[target_data.non_uniform_dim] == num_entries_output``

    Returns:
            As ``target_data``, with the values from ``source_data`` inserted according to the pairs of
            corresponding indices in ``source_indices`` and ``target_indices``.

            Shape in case of a tensor:

                ``(batch_size, num_entries_output, ...)``

            In case of a RaggedBatch, the following holds:

                - ``target_data_filled.shape[0] == batch_size``
                - ``target_data_filled.shape[target_data_filled.non_uniform_dim] == num_entries_output``

    Example:

        - '-' indicates values in the input which are not used in the mapping
        - '*' indicates filler values in `source_indices` and `target_indices`, which are ignored.
        - '..' indicates data which remains unchanged, i.e. is the same as in the `target_data` parameter
          and the output.

        Each depicted element in `source_data` and `target_data` may represent a single value
        (in case of 2D tensors), or itself be a non-scalar entry (in case that the data has more than 2 dimensions).

        .. image:: images/Mapping_ragged.png
            :alt: Illustration of the mapping operation
            :align: center

    """
    num_batch_dims = target_indices.non_uniform_dim
    assert (
        target_indices.dim() == num_batch_dims + 1 and source_indices.dim() == num_batch_dims + 1
    ), "Indices must have exactly one dimension in addition to the batch dimensions"
    assert (
        target_indices.shape[:num_batch_dims] == source_indices.shape[:num_batch_dims]
    ), "Batch shape mismatch"
    assert (
        target_indices.shape[num_batch_dims] == source_indices.shape[num_batch_dims]
    ), "Maximum number of indices mismatch"

    is_target_data_ragged_batch = isinstance(target_data, RaggedBatch)
    is_source_data_ragged_batch = isinstance(source_data, RaggedBatch)
    if is_target_data_ragged_batch:
        target_data_non_uniform_dim = target_data.non_uniform_dim
        target_data = target_data.get_non_uniform_dimension_transposed_to(num_batch_dims)
        target_data_tensor = target_data.tensor
    else:
        target_data_non_uniform_dim = 1
        target_data_tensor = target_data
    if is_source_data_ragged_batch:
        source_data = source_data.get_non_uniform_dimension_transposed_to(num_batch_dims)
        source_data = source_data.tensor

    # TODO: add check for consistent sample sizes (debug mode?)
    res = BatchedIndexMapping.apply(
        source_data,
        source_indices.tensor,
        target_indices.tensor,
        source_indices.sample_sizes,
        target_data_tensor,
    )
    if is_target_data_ragged_batch:
        res = target_data.create_with_sample_sizes_like_self(res, num_batch_dims)
        res = res.get_non_uniform_dimension_transposed_to(target_data_non_uniform_dim)
    else:
        if target_data_non_uniform_dim != 1:
            res = res.transpose(1, target_data_non_uniform_dim)

    return res
