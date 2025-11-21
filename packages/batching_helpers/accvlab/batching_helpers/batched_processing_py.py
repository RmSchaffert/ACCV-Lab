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

from typing import NamedTuple, Union, Sequence, List, Any, Optional
import torch

from .data_format import RaggedBatch

from .batched_bool_indexing import batched_bool_indexing


def average_over_targets(data: RaggedBatch, nans_to_zero: bool = True) -> torch.Tensor:
    """Average along the non-uniform dimension, considering only the valid entries.

    The dimension to average over is ``data.non_uniform_dim``.

    Args:
        data: Data to average
        nans_to_zero: Whether to replace NaNs with zeros after averaging. Default is ``True``.

    Returns:
        Tensor containing per-sample averages
    """
    data = data.get_non_uniform_dimension_transposed_to(data.num_batch_dims)
    masked_data = data.with_padded_set_to(0.0)
    summed = torch.sum(masked_data.tensor, dim=data.num_batch_dims, dtype=masked_data.tensor.dtype)

    # When dividing, make sure that the batch dimension corresponds to the innermost dimension.
    # This is so that sata.sample_sizes is broadcasted correctly.
    multi_dim = summed.dim() > 1
    if multi_dim:
        # Ensure that the batch dimensions are the innermost dimensions
        summed = summed.permute(*range(data.num_batch_dims, summed.dim()), *range(data.num_batch_dims))
    res = summed / data.sample_sizes
    if multi_dim:
        # Move batch dimensions back to their original positions
        res = res.permute(*range(-data.num_batch_dims, 0), *range(res.dim() - data.num_batch_dims))
    if nans_to_zero:
        res = torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    return res


def sum_over_targets(data: RaggedBatch) -> torch.Tensor:
    """Sum over the non-uniform dimension, considering only the valid entries.

    The dimension to average is ``data.non_uniform_dim``.

    Args:
        data: Data to average

    Returns:
        Tensor containing per-sample sums
    """
    masked_data = data.with_padded_set_to(0.0)
    summed = torch.sum(masked_data.tensor, dim=data.non_uniform_dim, dtype=masked_data.tensor.dtype)
    return summed


def apply_mask_to_tensor(data: torch.Tensor, mask: torch.Tensor, value_to_set: float = 0.0) -> torch.Tensor:
    """Apply mask to tensor

    Apply a mask to a tensor, setting any elements in the tensor corresponding to ``False`` entries in the mask
    to a fixed value. The mask may have fewer dimensions than the data. In this case, it is assumed to be constant in the
    remaining dimensions and the available dimensions correspond to the outer data dimensions (i.e. starting from ``dim==0``
    in the data).

    Args:
        data: Data to apply mask to
        mask: Mask to apply
        value_to_set: Value to set for padded entries. Default is 0.0

    Returns:
        Like ``data``, but with the mask applied
    """
    # Broadcast mask if possible
    num_extra_dims = data.dim() - mask.dim()
    if num_extra_dims > 0:
        extra_shape = [1] * num_extra_dims
        mask = torch.reshape(mask, (*mask.shape, *extra_shape))
    mask_broadcasted = torch.broadcast_to(mask, data.shape)
    # mask_squeezed = squeeze_except_batch_and_sample(mask)
    # Copy data to not modify the original tensor
    if mask.numel() > 0:
        res = data.clone()
        # Apply mask on copy
        res[torch.logical_not(mask_broadcasted)] = value_to_set
    else:
        res = data
    return res


def squeeze_except_batch_and_sample(
    data: Union[torch.Tensor, RaggedBatch],
) -> Union[torch.Tensor, RaggedBatch]:
    """Squeeze the data except the batch dimension and the non-uniform dimension representing the sample size.

    For tensors, the batch dimension is always ``dim==0``. The non-uniform dimensions is assumed to be
    ``dim==1``. For ragged batches, the batch dimensions are the first ``data.num_batch_dims`` dimensions and
    the non-uniform dimension is the ``data.non_uniform_dim`` dimension.

    This function is designed to preserve the batch and non-uniform dimensions, which have a special meaning,
    while allowing to squeeze away other dimensions.

    Important:
        Note that as a result of the squeezing, the non-uniform dimension may change to a different dimension.
        This happens if there are any dimensions before the non-uniform dimension which are squeezed away.
        For example:

        >>> example_batch.shape
        torch.Size([4, 1, 1, 3, 4])
        >>> example_batch.num_batch_dims
        2
        >>> example_batch.non_uniform_dim
        3
        >>> example_batch_squeezed = squeeze_except_batch_and_sample(example_batch)
        >>> example_batch_squeezed.shape
        torch.Size([4, 1, 3, 4])
        >>> example_batch_squeezed.non_uniform_dim
        2

        Note that the non-uniform dimension is now ``dim==2`` instead of ``dim==3``. Also, as ``dim==1`` is
        one of the batch dimensions, it is not squeezed away. The same would be true for
        any other batch dimension or the non-uniform dimension.

    Args:
        data: Data to be squeezed

    Returns:
        Squeezed data
    """

    def get_num_squeezed_dimensions_before(shape, num_batch_dims, non_uniform_dim):
        num_skipped = 0
        for s in shape[num_batch_dims:non_uniform_dim]:
            if s == 1:
                num_skipped += 1
        return num_skipped

    is_ragged_batch = isinstance(data, RaggedBatch)
    if is_ragged_batch:
        non_uniform_dim_orig = data.non_uniform_dim
        num_batch_dims = data.num_batch_dims
        # We cannot simply assume that the non-uniform dimension after squeezing is the dimension it was before, as dimensions
        # prior to it may be squeezed away. Therefore, check how many would be squeezed and adjust accordingly
        num_squeezed_dims_before_non_uniform = get_num_squeezed_dimensions_before(
            data.shape, data.num_batch_dims, non_uniform_dim_orig
        )
        data_to_use = data.tensor
        dims_to_squeeze = tuple(range(num_batch_dims, non_uniform_dim_orig)) + tuple(
            range(non_uniform_dim_orig + 1, data.dim())
        )
    else:
        num_batch_dims = 1
        data_to_use = data
        dims_to_squeeze = tuple(range(2, data_to_use.dim()))

    res = torch.squeeze(data_to_use, dims_to_squeeze)

    if is_ragged_batch:
        non_uniform_dim_new = non_uniform_dim_orig - num_squeezed_dims_before_non_uniform
        res = data.create_with_sample_sizes_like_self(res, non_uniform_dim=non_uniform_dim_new)

    return res


def get_compact_from_named_tuple(mask: torch.Tensor, data: NamedTuple) -> NamedTuple:
    """Get a compact version of all tensors (as :class:`RaggedBatch` instances) in a named tuple as a new named tuple of the same type.

    See :func:`get_compact_lists` for details on compactification and an illustration of example inputs & outputs.
    This function works in the same way, but using named tuples instead of plain sequences as inputs & outputs.

    Args:
        mask: Mask indicating which elements are valid (True) or not (False). Shape: (dim_size_0, dim_size_1)
        data: Named tuple containing tensors to be compactified. For each tensor, the shape is (dim_size_0, dim_size_1, ...).
            Note that different tensor in the list may have a different number of dimensions and a different size
            except for the first 2 dimensions, which have to correspond to the mask. The tuple may contain elements
            which are not tensors, and such elements will remain unchanged in the output.

    Returns:
        Compact data, where elements containing tensors in the input are replaces by :class:`RaggedBatch` instances,
        while elements of other types remain unchanged.
    """
    res_list = get_compact_lists(mask, data)
    DataType = type(data)
    res = DataType(*res_list)
    return res


def get_compact_lists(
    mask: torch.Tensor, data: Sequence[Union[torch.Tensor, Any]]
) -> List[Union[RaggedBatch, Any]]:
    """For a list of data tensors and a mask indicating which entries are valid, get a compactified version of the data.

    Compactification is performed along the second dimension (i.e. ``dim==1``). Compactification in this context means that
    for this dimension:

        - The size of the dimension is reduced so that it exactly fits the maximum number of valid elements over all samples,
          i.e. the size of the dimension is ``max(sum(mask, dim==1))``.
        - The data is converted to :class:`RaggedBatch` instances for further processing (see documentation of
          :class:`RaggedBatch` for details of the format)

    Args:
        mask: Mask indicating which elements are valid (True) and which are not (False).
            Shape: (dim_size_0, dim_size_1)
        data: Sequence (e.g. ``list``) of tensors to be compactified. For each tensor, the shape is
            (dim_size_0, dim_size_1, ...).
            Note that different tensor in the sequence may have a different number of dimensions and a different size
            except for the first 2 dimensions, which have to correspond to the mask. The sequence may contain elements
            which are not tensors, and such elements will remain unchanged in the output.

    Returns:
        Compact data, where elements containing lists of tensors in the input are replaces by :class:`RaggedBatch`
        instances, while other elements remain unchanged.

    Example:
        In the illustration below:
          - Letters indicate data entries that are indexed in the input (and therefore appear in the output)
          - '-' indicates entries where the actual values are not relevant (in the input).
          - '*' indicates filler values in :class:`RaggedBatch` instances.

        The illustration shows a single input entry ``data[d]``, where ``isinstance(data[d], torch.Tensor) == True``.
        Note that for non-tensor elements of ``data``, the data is not changed.

        .. image:: images/Compactification_ragged.png
            :alt: Illustration of the compactification operation
            :align: center

        Each depicted element in the data may represent a single value (in case of 2D tensors),
        or itself be a non-scalar entry (in case that the data has more than 2 dimensions).

    """
    num_data = len(data)
    mask = mask.bool()
    num_vals = mask.sum(dim=1, keepdims=True)
    max_num_vals = num_vals.max().item()
    num_samples = mask.shape[0]

    val_range = torch.arange(0, max_num_vals, device=mask.device)

    val_range_block = val_range.repeat((num_samples, 1))

    mask_res = val_range_block < num_vals

    # Flatten the mask and sample sizes to 1D tensors
    # Originally, `num_vals` is a 2D tensor with shape `(num_samples, 1)` in order to
    # broadcast correctly against the 2D `mask_res`.
    num_vals_1d = num_vals.flatten()

    res = [None] * num_data
    for i, el in enumerate(data):
        if isinstance(el, torch.Tensor):
            size_res = list(el.shape)
            if len(size_res) < 2:
                size_res = [size_res[0], None]
            size_res[1] = max_num_vals
            curr_res = torch.zeros(size_res, dtype=el.dtype, device=el.device)
            curr_res[mask_res] = el[mask]
            res[i] = RaggedBatch(curr_res, mask_res, num_vals_1d)
        else:
            res[i] = el
    return res


def combine_data(
    data_list: Sequence[Union[Sequence, torch.Tensor]],
    other_with_same_sample_sizes: RaggedBatch = None,
    device: Optional[Union[torch.device, str]] = None,
    flatten_batch_dims: bool = True,
) -> RaggedBatch:
    """Combine data given as an (optionally nested) sequence of tensors to a single RaggedBatch

    Nested sequences can be processed in two different ways:

        - If ``flatten_batch_dims`` is ``True``, the sequence is flattened as if it were a single sequence.
          In this case, the result is a single RaggedBatch instance with a single batch dimension.
          The sequence of the samples in the batch is ordered as they appear in ``data_list`` when
          traversed in depth-first order. In this case, there are no requirements on the length
          of the sequences in `data_list` on any nesting level and the number of elements in the
          sequences can vary between individual elements and from one nesting level to the next.
          Also, the number of nesting levels can vary between individual elements.
        
        - If ``flatten_batch_dims`` is ``False``, the sequence is treated as a nested sequence and the
          nesting levels are preserved as batch dimensions, i.e. each nesting level corresponds 
          to one batch dimension. As the batching dimensions need to be of uniform size, the number 
          of elements in all lists for a given nesting level needs to be identical. 
      
          For example, the following needs to be fulfilled for the 2nd nesting level:

          .. code-block:: python
  
              len(data_list[0][0]) == len(data_list[1][0]) == ... == len(data_list[n-1][0]) == \
              len(data_list[0][1]) == len(data_list[1][1]) == ... == len(data_list[n-1][1]) == \
              ... \
              len(data_list[0][m-1]) == len(data_list[1][m-1]) == ... == len(data_list[n-1][m-1])
        
          where ``n`` is the number of elements in the 1st nesting level and ``m`` is the number of elements 
          in the 2nd nesting level. 

    The individual tensors contained in ``data_list`` need to match in size except for the dimension 
    ``dim==0`` (which will correspond to the non-uniform dimension in the resulting :class:`RaggedBatch` 
    instance).

    Warning:
        If ``other_with_same_sample_sizes`` is provided, it is assumed that the batch shape and sample
        sizes are identical. If this is not the case, the behavior is undefined.

    Example:

        In the example below, the :func:`combine_data` operation is applied to a sequence of 4 tensors,
        each corresponding to a single sample. As there is no nesting, a single batch dimension is created.
        Note that in the image below:

          - Letters indicate data entries that are valid (i.e. correspond to the actual data).
          - '*' Indicates padded filler entries (i.e. invalid entries) in the data.

        .. image:: images/Combine_ragged.png
            :alt: Illustration of the combine data operation
            :align: center

        Each depicted element may represent a single value (corresponding to scalar data and 0 data dimensions),
        or itself represent a non-scalar entry (in case for one or more data dimensions).

    Args:
        data_list: Sequence of tensors to combine.
        other_with_same_sample_sizes: Other RaggedBatch instance with the same batch size and sample sizes. This is 
            optional and if provided, the mask and sample sizes tensors are shared between
            this RaggedBatch instance and the result, reducing the amount of needed memory.
        device: Device on which to create the resulting RaggedBatch. If not provided, 
            the device of the first element of ``data_list`` is used.
        flatten_batch_dims: Whether to flatten the batch dimensions (see discussion above for details). Default is ``True``.
    
    Returns:
        The combined data. 
        Shape:

            - If ``flatten_batch_dims`` is ``True``, the batch dimension is ``dim==0`` and the non-uniform size dimension is ``dim==1``. 
            - If ``flatten_batch_dims`` is ``False``, the batch dimensions correspond to the nesting levels
                of ``data_list``. The non-uniform size dimension is ``dim==num_batch_dims`` (i.e. the dimension
                immediately following the batch dimensions).
            - The remaining dimensions are as in the input tensors.
    """
    assert isinstance(data_list, Sequence), "`data_list` must be a sequence"
    assert len(data_list) > 0, "`data_list` must not be empty"
    reuse_mask_and_sample_sizes = other_with_same_sample_sizes is not None

    def process_with_flattening(data_list):

        nonlocal device

        def get_first_nonempty_element(data_list):
            for el in data_list:
                if el.numel() > 0:
                    return el
            return None

        # TODO: move max_numel computation to a separate function & remove the function nesting here
        def flatten_batch_dims_get_orig_dims_and_max_numel(data):
            def flatten_recurrsion(data):
                is_tensor = isinstance(data, torch.Tensor)
                if is_tensor:
                    res = [data]
                    max_numel = data.shape[0]
                elif isinstance(data, Sequence) and not isinstance(data, str):
                    res = []
                    max_numel = 0
                    for element in data:
                        res_to_add, max_numel_inner = flatten_recurrsion(element)
                        if max_numel_inner > max_numel:
                            max_numel = max_numel_inner
                        res += res_to_add
                else:
                    raise AssertionError(
                        "The data to combine must be a tensor or a (nested) sequence of tensors. "
                        f"Got {type(data)}."
                    )
                return res, max_numel

            res, max_numel = flatten_recurrsion(data)
            return res, max_numel

        # Original implementation for flatten_batch_dims=True
        flattened, max_numel = flatten_batch_dims_get_orig_dims_and_max_numel(data_list)
        num_flattened = len(flattened)
        dims_mask = (num_flattened, max_numel)
        # Empty elements may not have the same number of dimensions as elements containing data.
        # Therefore, do not use empty elements as examples if there are any non-empty elements available.
        sample_element = get_first_nonempty_element(flattened)
        # If there are no non-empty elements, use an empty element as a sample element.
        # In this case, we can still infer the batch size from the number of elements.
        if sample_element is None and num_flattened > 0:
            sample_element = flattened[0]
        if device is None:
            device = sample_element.device if sample_element is not None else None
        # If there are no elements at all, create an empty RaggedBatch instance
        if sample_element is None:
            res = RaggedBatch.Empty(2, 1, device=device)
            return res

        dims_data_remaining_inner = tuple(sample_element.shape[1:])
        dims_data = dims_mask + dims_data_remaining_inner
        data = torch.zeros(dims_data, dtype=sample_element.dtype, device=device)
        if not reuse_mask_and_sample_sizes:
            nums_vals = torch.empty(num_flattened, dtype=torch.int64, device="cpu")

        for i, el in enumerate(flattened):
            size_elem = min(el.shape[0], el.numel())
            if not reuse_mask_and_sample_sizes:
                nums_vals[i] = size_elem
            # Note that this check is potentially needed for the data, as empty elements may have a number of dimensions different from
            # other elements, leading to errors when trying to fill the data
            if size_elem > 0:
                data[i, 0:size_elem, ...] = el

        if not reuse_mask_and_sample_sizes:
            nums_vals = nums_vals.to(device=device)
            res = RaggedBatch(data, sample_sizes=nums_vals)
        else:
            assert (
                num_flattened == other_with_same_sample_sizes.sample_sizes.shape[0]
            ), "Number of samples does not match `other_with_same_sample_sizes`"
            assert (
                dims_mask == other_with_same_sample_sizes.mask.shape
            ), "Needed mask dimension does not match `other_with_same_sample_sizes`"
            res = other_with_same_sample_sizes.create_with_sample_sizes_like_self(
                data, non_uniform_dim=1, device=device
            )

        return res

    def process_as_nested(data_list):

        nonlocal device

        # Get the first contained tensor from a nested sequence
        def get_first_element(data_list):
            if isinstance(data_list, torch.Tensor):
                return data_list
            else:
                return get_first_element(data_list[0])

        # Determine the batch shape and validate consistency
        def get_batch_shape(data, level=0):
            assert isinstance(data, Sequence) and not isinstance(data, str), "`data` must be a sequence"
            # Check first element to determine type for this level
            first_item = data[0]
            if isinstance(first_item, torch.Tensor):
                # All elements at this level must be tensors
                for item in data[1:]:
                    if not isinstance(item, torch.Tensor):
                        raise AssertionError(
                            "The data to combine must be a tensor or a (nested) sequence of tensors. "
                            f"Got {type(item)} contained in the sequence at level {len(batch_shape)}."
                        )
                batch_shape = [len(data)]
            else:
                # Get shape from first sequence
                sub_shape = get_batch_shape(first_item, level + 1)
                # Verify consistency for remaining sequences
                for item in data[1:]:
                    if not isinstance(item, Sequence) or isinstance(item, str):
                        raise AssertionError(
                            "The data to combine must be a tensor or a (nested) sequence of tensors. "
                            f"Got {type(item)} contained in the sequence at level {level + 1}."
                        )
                    this_sub_shape = get_batch_shape(item, level + 1)
                    if this_sub_shape != sub_shape:
                        raise AssertionError(
                            f"Inconsistent sequence length structure at level {level + 1}. "
                            f"Expected shape {sub_shape}, got {this_sub_shape}"
                        )

                batch_shape = [len(data)] + sub_shape

            return batch_shape

        # Find maximum size along the first dimension of all tensors in the nested structure
        def find_max_numel(data):
            if isinstance(data, torch.Tensor):
                return data.shape[0]
            else:
                max_numel = 0
                for d in data:
                    max_numel = max(max_numel, find_max_numel(d))
                return max_numel

        # Fill the nested data and sample sizes (sample_sizes is None if generate_mask_and_sample_sizes is False)
        def fill_data_and_sample_sizes(data_seq, data_tensor, sizes_tensor, batch_indices=()):
            if isinstance(data_seq, torch.Tensor):
                size_elem = min(data_seq.shape[0], data_seq.numel())
                if size_elem > 0:
                    # Describe the region of the data tensor to fill
                    idx = batch_indices + (slice(0, size_elem),) + (Ellipsis,)
                    # Fill the defined region of the data tensor with the input tensor ("leaf node" of the nested input sequence)
                    data_tensor[idx] = data_seq
                    # Set sample size (if needed)
                    if sizes_tensor is not None:
                        sizes_tensor[batch_indices] = size_elem
            else:
                # If the current element is not a "leaf node", continue the recursion
                for i, item in enumerate(data_seq):
                    new_indices = batch_indices + (i,)
                    fill_data_and_sample_sizes(item, data_tensor, sizes_tensor, new_indices)

        # Get batch shape
        batch_shape = get_batch_shape(data_list)
        num_batch_dims = len(batch_shape)

        # Find the maximum size of the non-uniform dimension across all tensors
        if not reuse_mask_and_sample_sizes:
            max_numel = find_max_numel(data_list)
        else:
            max_numel = other_with_same_sample_sizes.mask.shape[-1]

        # Get a sample element to determine the data type and remaining dimensions
        sample_element = get_first_element(data_list)
        if sample_element is None or max_numel == 0:
            # TODO: check if this is needed
            # Handle empty case
            device = torch.device("cpu") if device is None else device
            data = torch.empty((*batch_shape, 0), dtype=torch.float32, device=device)
            mask = torch.empty((*batch_shape, 0), dtype=torch.bool, device=device)
            sample_sizes = torch.zeros(batch_shape, dtype=torch.int64, device=device)
            res = RaggedBatch(data, mask, sample_sizes, non_uniform_dim=num_batch_dims)
            return res

        if device is None:
            device = sample_element.device

        # Create the data tensor with the correct batch dimensions
        dims_data_remaining_inner = tuple(sample_element.shape[1:])
        dims_data = (*batch_shape, max_numel) + dims_data_remaining_inner
        data = torch.zeros(dims_data, dtype=sample_element.dtype, device=device)

        # Create sample sizes tensor (if needed)
        if not reuse_mask_and_sample_sizes:
            sample_sizes = torch.zeros(batch_shape, dtype=torch.int64, device=device)
        else:
            sample_sizes = None

        # Fill the data and sample sizes (the latter only if needed)
        fill_data_and_sample_sizes(data_list, data, sample_sizes)
        if not reuse_mask_and_sample_sizes:
            sample_sizes = sample_sizes.to(device=device)

        if not reuse_mask_and_sample_sizes:
            res = RaggedBatch(data, sample_sizes=sample_sizes, non_uniform_dim=num_batch_dims)
        else:
            # Use other_with_same_sample_sizes
            assert other_with_same_sample_sizes.sample_sizes.shape == tuple(
                batch_shape
            ), "Sample sizes shape does not match required batch shape"
            res = other_with_same_sample_sizes.create_with_sample_sizes_like_self(
                data, non_uniform_dim=num_batch_dims, device=device
            )

        return res

    if flatten_batch_dims:
        res = process_with_flattening(data_list)
    else:
        res = process_as_nested(data_list)

    return res


def get_indices_from_mask(mask: Union[torch.Tensor, RaggedBatch]) -> RaggedBatch:
    """Get the indices from a mask.

    :gpu:

    For each sample, the indices correspond to the elements in the mask that are ``True``.

    This functionality is e.g. useful when boolean indexing needs to be applied multiple times for different
    data and the same mask, as the indexing using numerical indices (using :func:`batched_indexing_access`) is
    more efficient than boolean indexing (using :func:`batched_bool_indexing`). Note that for a single
    application of boolean indexing, the :func:`batched_bool_indexing` function is more efficient.

    Note:
        Only 2D masks (batch_size, num_elements) are supported.

    See also:
        :func:`batched_mask_from_indices` :func:`batched_indexing_access` :func:`batched_bool_indexing`

    Args:
        mask: The mask to get the indices from.

    Returns:
        The indices from the mask.

    Example:

        In the illustration below, '*' indicates invalid indices, i.e. padding to make the tensor uniform for
        samples where the number of indices is smaller than ``max_num_indices``.

        .. image:: images/IndicesFromMask_ragged.png
            :alt: Illustration of the indices from mask operation
            :align: center

    """

    if isinstance(mask, RaggedBatch):
        assert (
            mask.num_batch_dims == 1
        ), "Only RaggedBatch instances with a single batch dimension are supported"
        mask = mask.with_padded_set_to(False)
        mask = mask.tensor

    assert mask.ndim == 2, "Only 2D masks (batch_size, num_elements) are supported"

    batch_size = mask.shape[0]
    num_elements = mask.shape[1]

    indices_all = torch.arange(num_elements, device=mask.device, dtype=torch.int64)
    indices_all = indices_all.unsqueeze(0).expand(batch_size, -1)
    indices = batched_bool_indexing(indices_all, mask)

    return indices
