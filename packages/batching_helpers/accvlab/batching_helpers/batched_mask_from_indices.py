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
from accvlab.batching_helpers.batched_processing_py import RaggedBatch
import accvlab.batching_helpers.batched_indexing_access_cuda as batched_indexing_access_cuda


def get_mask_from_indices(mask_num_targets: int, indices: RaggedBatch) -> torch.Tensor:
    """Get a mask from indices, where the indices indicate which elements in the mask should be ``True``.

    :gpu:

    The indices for each sample define the ``True`` values in the mask for that sample
    (i.e. the corresponding row in the mask).

    For each sample ``i``, the operation performed by this function is equivalent to:
        ``mask[i, indices.tensor[i, :indices.sample_sizes[i]]] = True``

    Please also see the documentation of :class:`RaggedBatch` for more details on the format of the indices,
    including the ``tensor`` and ``sample_sizes`` attributes.

    Note:

        This function is not the inverse of :func:`get_indices_from_mask`, as the index order is not
        preserved when converting from indices to a mask.

    Args:
        mask_num_targets: The number of targets in the mask, i.e. the ``mask.shape[1]`` to use
        indices: For each sample (element along the batch dimension), the indices of elements to set to ``True``.
            Shape: (batch_size, max_num_indices)

    Returns:
        Resulting mask. Shape: (batch_size, ``num_targets_in_mask``)

    Example:

        In the illustration below, '*' indicates invalid indices, i.e. padding to make the tensor uniform for
        samples where the number of indices is smaller than ``max_num_indices``. Note that the index order
        does not matter for the resulting mask.

        .. image:: images/MaskFromIndices_ragged.png
            :alt: Illustration of the mask from indices operation
            :align: center

    """
    data = indices.tensor.contiguous()
    sample_sizes = indices.sample_sizes.contiguous()
    res = batched_indexing_access_cuda.get_mask_from_indices(data, sample_sizes, mask_num_targets)
    return res
