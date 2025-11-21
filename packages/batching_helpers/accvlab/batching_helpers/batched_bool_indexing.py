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

from typing import Union
import torch

from .data_format import RaggedBatch


def _compare_indexed_data_and_mask(
    data: Union[RaggedBatch, torch.Tensor],
    mask: Union[RaggedBatch, torch.Tensor],
):
    '''
    Compare the data and mask to ensure that they are compatible.

    Note that if both the data and mask are RaggedBatch instances, it is assumed that the sample sizes match.
    Only the maximum sample size is checked, not the individual sample sizes.

    Raises:
        AssertionError: If the data and mask are not compatible.
    '''
    is_data_ragged = isinstance(data, RaggedBatch)
    is_mask_ragged = isinstance(mask, RaggedBatch)

    if is_data_ragged and is_mask_ragged:
        assert (
            data.num_batch_dims == mask.num_batch_dims
        ), "Data and mask must have the same number of batch dimensions"
        assert data.batch_shape == mask.batch_shape, "Data and mask must have the same batch shape"
        assert (
            data.max_sample_size == mask.max_sample_size
        ), "Data and mask must have the same maximum sample size"
    elif is_data_ragged:
        assert data.num_batch_dims == 1, "Data must have exactly one batch dimension if mask is a tensor"
        assert data.batch_shape[0] == mask.shape[0], "Data and mask must have the same number of samples"
        assert (
            data.max_sample_size == mask.shape[1]
        ), "Maximum sample size of data must correspond to `input_mask.shape[1]` if the mask is a tensor"
    elif is_mask_ragged:
        assert (
            mask.num_batch_dims == 1
        ), "Mask must have exactly one batch dimension if input data is a tensor"
        assert mask.batch_shape[0] == data.shape[0], "Mask and data must have the same number of samples"
        assert (
            mask.max_sample_size == data.shape[1]
        ), "Maximum sample size of mask must correspond to `input_data.shape[1]` if the input data is a tensor"
    else:
        assert data.shape[0] == mask.shape[0], "Data and mask must have the same number of samples"
        assert data.shape[1] == mask.shape[1], "Data and mask must have the same maximum sample size"


def _mask_the_mask(
    mask: Union[RaggedBatch, torch.Tensor], data: Union[RaggedBatch, torch.Tensor]
) -> Union[RaggedBatch, torch.Tensor]:
    '''Mask the mask to ensure that filler elements are set to False.

    If the mask is a RaggedBatch, we ensure that filler elements are set to False.

    If the mask is a tensor and the data is a RaggedBatch, we assume that the mask has the same sample sizes
    as the data. Therefore, we perform a masking of the mask with ``data.mask``.

    If both the mask and the data are tensors, the mask is not modified.

    Args:
        mask: The mask to apply masking to.
        data: The data which is indexed using the mask.

    Returns:
        The masked mask.
    '''
    if isinstance(mask, RaggedBatch):
        mask = mask.with_padded_set_to(False)
    elif isinstance(data, RaggedBatch):
        mask = mask * data.mask
    return mask


def batched_bool_indexing(
    input_data: Union[RaggedBatch, torch.Tensor],
    input_mask: Union[RaggedBatch, torch.Tensor],
) -> RaggedBatch:
    """
    Batched boolean indexing.

    This function performs batched boolean indexing on the input data using the input mask.
    Both the input data and the input mask can be either :class:`RaggedBatch` or :class:`torch.Tensor`
    instances.

    The indexing is performed along the non-uniform dimension of the input data. For tensors,
    the non-uniform dimension is assumed to be ``dim==1``.

    In case that one ``input_data`` or ``input_mask`` is a :class:`torch.Tensor` and the other is
    a :class:`RaggedBatch`:

      - A single batch dimension must be used
      - The sample sizes of the :class:`RaggedBatch` are assumed to also apply to the tensor
      - The non-uniform dimension of the tensor is assumed to be ``dim==1``

    If both the input data and the input mask are tensors:

      - All entries along ``dim==1`` (the non-uniform dimension) are assumed to be valid (i.e. sample size for
        each sample corresponds to the size of this dimension)
      - The output will also be a :class:`RaggedBatch` in this case (as in general, the number of ``True``
        values in the mask is not the same for all samples)

        - A single batch dimension will be used (consistent to the assumption about the input data)
        - The non-uniform dimension will be at ``dim==1`` (consistent to the assumption about the input data)

    If both the input data and the input mask are :class:`RaggedBatch` instances, multiple batch dimensions
    are supported.

    Warning:
        If both the input data and mask are :class:`RaggedBatch` instances, it is assumed that the sample
        sizes match. Only the maximum sample size is checked and if the individual sample sizes are not the
        same, the behavior is undefined.

    Args:
        input_data: The data to index into.
            Shape (in case of the non-uniform dimension being ``dim==1``):
            ``(*batch_shape, max_sample_size, *data_shape)``,
            where ``max_sample_size`` is the maximum sample size of the input data.
            Note that the data_shape may contain 0 or more entries.
            If the non-uniform dimension is not ``dim==1``, the ``max_sample_size`` is also not the size of the
            second dimension (``dim==1``), but of the corresponding dimension.
        input_mask: The mask to use for indexing.
            Shape: ``(*batch_shape, max_sample_size)``,
            where ``max_sample_size`` is the maximum sample size of the input data.
            Note that ``data_shape`` is not present, as each data entry is treated as a single element
            in the indexing operation.

    Returns:
        :class:`RaggedBatch` instance containing the indexed data for each sample.

    Example:

        In the illustration below:
          - Letters indicate data entries that are indexed in the input (and therefore appear in the output)
          - '-' indicates entries where the actual values are not relevant (in the input).
          - '*' indicates filler values in :class:`RaggedBatch` instances.

        .. image:: images/BatchedBoolIndexAccess_ragged.png
            :alt: Illustration of the batched boolean indexing operation
            :align: center

        Each depicted entry in ``input_data`` may represent a single value (in case of
        2D tensors), or itself be a non-scalar entry (in case that ``input_data`` has more than 2 dimensions).
        The entries in ``input_mask`` are always scalar.
    """

    _compare_indexed_data_and_mask(input_data, input_mask)

    is_input_data_ragged = isinstance(input_data, RaggedBatch)
    is_mask_ragged = isinstance(input_mask, RaggedBatch)

    # Get the mask to use for the result (with filler elements set to False, either using the mask itself
    # (if it is a RaggedBatch) or using the mask of the to_write (if it is a RaggedBatch and the mask is a
    # tensor))
    input_mask = _mask_the_mask(input_mask, input_data)

    if not is_input_data_ragged:
        batch_shape = torch.Size([input_data.shape[0]])
        is_multi_batch_dim = False
        num_batch_dims = 1
    else:
        batch_shape = input_data.batch_shape
        is_multi_batch_dim = input_data.num_batch_dims > 1
        num_batch_dims = input_data.num_batch_dims

    # Convert the data to the correct format
    if is_input_data_ragged:
        orig_input_data_non_uniform_dim = input_data.non_uniform_dim
        if num_batch_dims > 1:
            input_data = input_data.flatten_batch_dims()
        input_data = input_data.get_non_uniform_dimension_transposed_to(1)
    if is_mask_ragged:
        # Note that for the mask, we do not need to transpose the non-uniform dimension, as it is already
        # in the correct position (as it has no data dimensions).
        if num_batch_dims > 1:
            input_mask = input_mask.flatten_batch_dims()
        input_mask = input_mask.tensor

    # Get the sample sizes of the input data
    sample_sizes = input_mask.sum(dim=1, keepdim=False)

    # Create the output RaggedBatch
    max_sample_size = int(sample_sizes.max().item())
    batch_size = input_data.shape[0]
    output_data = RaggedBatch(
        torch.full(
            (batch_size, max_sample_size, *input_data.shape[2:]),
            0.0,
            dtype=input_data.dtype,
            device=input_data.device,
        ),
        sample_sizes=sample_sizes,
        non_uniform_dim=1,
    )

    # Now, we have a mask for both the input and output data (the latter will be generated from the
    # sample_sizes used to construct the output RaggedBatch)
    # We can use both masks to fill the output data with the input data. Note that while the masks are
    # different, the element correspondence is preserved as for each sample, the number of selected
    # element in the input is the same as the sample size of the outputs (by construction).
    def fill_output_data(tensor: torch.Tensor, mask: torch.Tensor):
        input_data_tensor = input_data.tensor if is_input_data_ragged else input_data
        tensor[mask] = input_data_tensor[input_mask]
        return tensor

    output_data = output_data.apply(fill_output_data)

    # If the input was a tensor, we do not need to change the non-uniform dimension or batch shape, as in
    # this case, the input data is expected to corresponf to the format that the output RaggedBatch already
    # has. Otherwise, we need to transpose the non-uniform dimension back to its original position and reshape
    # the batch dimensions back to the original shape.
    if is_input_data_ragged:
        # Reshape the batch dimensions back to the original shape
        if is_multi_batch_dim:
            output_data = output_data.reshape_batch_dims(batch_shape)
        # Transpose the non-uniform dimension back to its original position
        output_data = output_data.get_non_uniform_dimension_transposed_to(orig_input_data_non_uniform_dim)

    return output_data


def batched_bool_indexing_write(
    to_write: RaggedBatch,
    output_mask: Union[RaggedBatch, torch.Tensor],
    to_write_into: Union[RaggedBatch, torch.Tensor],
) -> Union[RaggedBatch, torch.Tensor]:
    """
    Batched boolean indexing write (inverse operation of batched_bool_indexing).

    This function performs the inverse operation of :func:`batched_bool_indexing`. It writes data from a
    :class:`RaggedBatch` into a target tensor or :class:`RaggedBatch` using a boolean mask to specify
    where to write the data.

    The writing is performed along the non-uniform dimension of `to_write_into`. For tensors,
    the non-uniform dimension is assumed to be `dim==1`.

    In case that one `output_mask` or `to_write_into` is a :class:`torch.Tensor` and the other is a
    :class:`RaggedBatch`:

      - A single batch dimension must be used
      - The sample sizes of the :class:`RaggedBatch` are assumed to also apply to the tensor (regardless of
        which of the two is which)
      - The non-uniform dimension of the tensor is assumed to be `dim==1`

    If both `output_mask` and `to_write_into` are tensors, all entries along `dim==1` (the non-uniform
    dimension) are assumed to be valid (i.e. sample size for each sample corresponds to the size of this
    dimension).

    Multiple batch dimensions are only supported if both `output_mask` and `to_write_into` are
    :class:`RaggedBatch` instances.

    Warning:
        If both `output_mask` and `to_write_into` are :class:`RaggedBatch` instances, it is assumed that
        the sample sizes match. Only the maximum sample size is checked and if the individual sample sizes are
        not the same, the behavior is undefined.

    Args:
        to_write: The :class:`RaggedBatch` containing the data to write.
            Shape (in case of the non-uniform dimension being `dim==1`):
            `(*batch_shape, max_sample_size, *data_shape)`,
            where `max_sample_size` is the maximum sample size of the data to write.
            Note that the data_shape may contain 0 or more entries.
            If the non-uniform dimension is not `dim==1`, the `max_sample_size` is also not the size of the
            second dimension (`dim==1`), but of the corresponding dimension.
        output_mask: The mask specifying where to write the data.
            Shape: `(*batch_shape, max_sample_size)`,
            Note that `data_shape` is not present, as each data entry is treated as a single element
            in the writing operation.
        to_write_into: The target tensor or :class:`RaggedBatch` to write into.
            Shape: `(*batch_shape, max_sample_size, *data_shape)`,
            The data_shape must match the data_shape of `to_write`.
            If the non-uniform dimension is not `dim==1`, the `max_sample_size` is also not the size of the
            second dimension (`dim==1`), but of the corresponding dimension.

    Returns:
        :class:`RaggedBatch` or :class:`torch.Tensor` instance containing the target data with the
        selected elements from `to_write` written into the positions specified by `output_mask`.

    Example:

        In the illustration below:
          - Letters indicate data entries that are indexed in the input (and therefore appear in the output)
          - '*' indicates filler values in :class:`RaggedBatch` instances.
          - '..' indicates data which remains unchanged, i.e. is the same as in the `to_write_into` parameter
            and the output.

        .. image:: images/BatchedBoolIndexWrite_ragged.png
            :alt: Illustration of the batched boolean indexing write operation
            :align: center

        Each depicted entry in `to_write` and `to_write_into` may represent a single value (in case of
        2D tensors), or itself be a non-scalar entry (in case that the data has more than 2 dimensions).
        The entries in `output_mask` are always scalar.
    """
    # Check the input
    assert isinstance(to_write, RaggedBatch), "to_write must be a RaggedBatch"
    _compare_indexed_data_and_mask(to_write_into, output_mask)

    # Check that the mask is a boolean mask
    is_mask_ragged = isinstance(output_mask, RaggedBatch)
    is_to_write_into_ragged = isinstance(to_write_into, RaggedBatch)

    # Get the mask to use for the result (with filler elements set to False, either using the mask itself
    # (if it is a RaggedBatch) or using the mask of the to_write (if it is a RaggedBatch and the mask is a
    # tensor))
    output_mask = _mask_the_mask(output_mask, to_write_into)

    # Get batch info
    batch_shape = to_write.batch_shape
    is_multi_batch_dim = to_write.num_batch_dims > 1
    num_batch_dims = to_write.num_batch_dims

    assert to_write.batch_shape == batch_shape, "to_write must have the same batch shape as the other inputs"
    assert (
        to_write.dim() == to_write_into.dim()
    ), "to_write and to_write_into must have the same number of dimensions"

    # Convert the data to the correct format
    if num_batch_dims > 1:
        to_write = to_write.flatten_batch_dims()
    to_write = to_write.get_non_uniform_dimension_transposed_to(1)
    if is_to_write_into_ragged:
        orig_to_write_into_non_uniform_dim = to_write_into.non_uniform_dim
        if num_batch_dims > 1:
            to_write_into = to_write_into.flatten_batch_dims()
        to_write_into = to_write_into.get_non_uniform_dimension_transposed_to(1)
    if is_mask_ragged:
        # Note that for the mask, we do not need to transpose the non-uniform dimension, as it is already
        # in the correct position (as it has no data dimensions).
        if num_batch_dims > 1:
            output_mask = output_mask.flatten_batch_dims()
        output_mask = output_mask.tensor

    # Get the result
    def apply_write(tensor: torch.Tensor):
        res = tensor.clone()
        res[output_mask] = to_write.tensor[to_write.mask]
        return res

    if is_to_write_into_ragged:
        res = to_write_into.apply(apply_write)
    else:
        res = to_write_into.clone()
        res[output_mask] = to_write.tensor[to_write.mask]

    # If to_write_into is a RaggedBatch, we need to adjust the format (i.e. the batch shape and the
    # non-uniform dimension) back to the original format
    if is_to_write_into_ragged:
        if is_multi_batch_dim:
            res = res.reshape_batch_dims(batch_shape)
        res = res.get_non_uniform_dimension_transposed_to(orig_to_write_into_non_uniform_dim)

    return res
