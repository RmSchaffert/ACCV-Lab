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

from typing import Optional

from nvidia.dali import types

from ..pipeline import SampleDataGroup

from .callable_base import CallableBase
from .sampler_base import SamplerBase
from .data_provider import DataProvider

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class SamplerInputCallable(CallableBase):
    '''Input callable using a sampler to provide data according to the sampler (also see :class:`~accvlab.dali_pipeline_framework.inputs.SamplerBase`).

    This callable also handles indicating the end of an epoch (by raising :class:`StopIteration`).
    Information on when an epoch ends is obtained from the sampler (which in turn should indicate this
    by raising :class:`StopIteration`, see documentation of :class:`SamplerBase`).

    As the sampler can have an internal state (while the callable is expected to be stateless), a look-up
    table is pre-generated at construction, leading to overhead and the need to know the maximum number of
    iterations in advance.

    Note:
        To avoid the overhead of pre-generating the look-up table, it is recommended to only use this class
        if a single process for data loading is not enough and prefer
        :class:`~accvlab.dali_pipeline_framework.inputs.SamplerInputIterable` in general.
    '''

    def __init__(
        self,
        data_provider: DataProvider,
        sampler: SamplerBase,
        max_num_iterations: int,
        pre_fetch_queue_length: int,
        shard_id: int = 0,
        num_shards: int = 1,
    ):
        '''
        Args:
            data_provider: Data provider to use (following the interface defined in :class:`DataProvider`).
            sampler: Sampler to use (following the interface defined in :class:`SamplerBase`).
            max_num_iterations: Maximum number of iterations that will be performed.
            pre_fetch_queue_length: Length of the pre-fetch queue depth of the DALI pipeline using this input callable. Used together with ``max_num_iterations`` to ensure that
                the sampling look-up table is large enough.
            shard_id: Shard ID (default value of 0 should be used if sharding is not used)
            num_shards: Total of shards (default value of 1 should be used if sharding is not used)
        '''

        self._data_provider = data_provider

        self._shard_id = shard_id
        self._num_shards = num_shards
        self._max_num_iterations = max_num_iterations
        self._pre_fetch_queue_length = pre_fetch_queue_length
        self._max_num_iterations_inc_queue = max_num_iterations + pre_fetch_queue_length

        self._look_up_table = []

        i = 0
        curr_epoch_look_up_table = []
        while i < self._max_num_iterations_inc_queue:
            try:
                batch = sampler.get_next_batch_indices()
                curr_epoch_look_up_table.append(batch)
                i += 1
            except StopIteration:
                self._look_up_table.append(curr_epoch_look_up_table)
                curr_epoch_look_up_table = []
                sampler.reset()
        self._look_up_table.append(curr_epoch_look_up_table)

        self._total_batch_size = len(self._look_up_table[0][0])
        self._local_batch_size = self._total_batch_size // num_shards

        assert (
            self._local_batch_size * self._num_shards == self._total_batch_size
        ), f"Total batch size ({self._total_batch_size}) not divisible by number of shards ({self._num_shards})."

    @property
    @override
    def used_sample_data_structure(self) -> SampleDataGroup:
        '''Data format blueprint used for the individual samples'''
        res = self._data_provider.sample_data_structure
        res.set_apply_mapping(False)
        return res

    @override
    def __call__(self, sample_info: types.SampleInfo) -> tuple:

        epoch_idx = sample_info.epoch_idx
        batch_idx = sample_info.idx_in_epoch // self._local_batch_size
        idx_in_local_batch = sample_info.idx_in_batch

        if epoch_idx >= len(self._look_up_table):
            raise RuntimeError(
                f"Maximum iteration number or pre-fetch queue length exceeded. SamplerInputCallable can only be used for the maximum number of iterations defined at construction\n"
                + f"({self._max_num_iterations} in this case) and a maximum pre-fetch queue depth defined at construction ({self._pre_fetch_queue_length} in this case). These two values\n"
                + f"define the total maximum number of batches which the callable can provide ({self._max_num_iterations_inc_queue} in this case)."
            )

        epoch_size = len(self._look_up_table[epoch_idx])

        if batch_idx >= epoch_size:
            raise StopIteration

        batch_of_indices = self._look_up_table[epoch_idx][batch_idx]

        idx_in_full_batch = idx_in_local_batch + self._shard_id * self._local_batch_size

        index_to_use = batch_of_indices[idx_in_full_batch]

        sample_data = self._data_provider.get_data(index_to_use)

        return sample_data.get_data()

    @property
    @override
    def length(self) -> Optional[int]:
        '''Number of batches in one epoch.

        If the underlying sampler is not epoch-based, the length is the overall number of batches
        that can be generated (i.e. the maximum number of iterations defined at construction plus
        the pre-fetch queue length).
        '''
        return len(self._look_up_table[0])
