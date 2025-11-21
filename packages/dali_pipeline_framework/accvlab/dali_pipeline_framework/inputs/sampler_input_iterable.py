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
from typing import Any, Sequence, Optional

from nvidia.dali import types

from ..pipeline import SampleDataGroup

from .iterable_base import IterableBase
from .sampler_base import SamplerBase
from .data_provider import DataProvider

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class SamplerInputIterable(IterableBase):
    '''Input iterable using a sampler to provide batches according to the sampler (also see :class:`SamplerBase`).

    The iterable can be used with a parallel external source. However, in this case, the data reading is
    performed in one worker process due to serial nature of an iterable. This means that while the data
    reading is asynchronous to the main thread, it is not further parallelized.

    This iterable also handles indicating the end of an epoch (by raising :class:`StopIteration`). Information
    on when an epoch ends is obtained from the sampler (which in turn should indicate this by raising
    :class:`StopIteration`, see documentation of :class:`SamplerBase`).
    After the end of the epoch, the iterable needs to be reset (by obtaining a new iterator) before data for
    the next epoch can be obtained.

    Note:
        If further parallelization is desired (i.e. more than one worker thread),
        :class:`SamplerInputCallable` can be used instead of this class (at the cost of pre-computing look-up
        tables in advance, see the corresponding note in the documentation of :class:`SamplerInputCallable`).
    '''

    def __init__(
        self,
        data_provider: DataProvider,
        sampler: SamplerBase,
        shard_id: int = 0,
        num_shards: int = 1,
    ):
        '''

        Args:
            data_provider: Data provider to use (following the interface defined in :class:`DataProvider`).
            sampler: Sampler to use (following the interface defined in :class:`SamplerBase`).
            shard_id: Shard ID (default value of 0 should be used if sharding is not used).
            num_shards: Total of shards (default value of 1 should be used if sharding is not used).
        '''

        self._data_provider = data_provider

        self._shard_id = shard_id
        self._num_shards = num_shards
        self._sampler = sampler

        self._local_batch_size = None
        self._total_batch_size = None
        self._epoch = 0

        self._before_first_iter_called = True
        self._sharding_set_up = False

    @property
    @override
    def used_sample_data_structure(self) -> SampleDataGroup:
        '''Data format blueprint used for the individual samples'''
        res = self._data_provider.sample_data_structure
        res.set_apply_mapping(False)
        return res

    @override
    def __iter__(self) -> 'SamplerInputIterable':
        if self._before_first_iter_called:
            self._before_first_iter_called = False
        else:
            if self._sampler.is_epoch_based:
                self._sampler.reset()
        return self

    @override
    def __next__(self) -> tuple:

        batch_indices = self._sampler.get_next_batch_indices()

        if not self._sharding_set_up:
            self._total_batch_size = len(batch_indices)
            self._local_batch_size = self._total_batch_size // self._num_shards
            assert (
                self._local_batch_size * self._num_shards == self._total_batch_size
            ), "Total batch size is not divisible by the number of used shards."
            self._sharding_set_up = True

        min_index_in_total_batch = self._shard_id * self._local_batch_size

        indices_to_use = batch_indices[
            min_index_in_total_batch : min_index_in_total_batch + self._local_batch_size
        ]

        sample_data = [self._data_provider.get_data(idx) for idx in indices_to_use]

        batch_res_data = [sd.get_data() for sd in sample_data]

        res = self._combine_res_to_batch(batch_res_data)

        return res

    @property
    @override
    def length(self) -> Optional[int]:
        '''Number of batches in one epoch.

        If the underlying sampler is not epoch-based, ``None`` is returned.
        '''
        return self._sampler.length

    @staticmethod
    def _combine_res_to_batch(per_sample_res: Sequence[Sequence[Any]]) -> Sequence[Sequence[Any]]:
        '''Combine per-sample results into a batch format.

        Transposes the data structure from ``per_sample_res[sample_idx][field_idx]`` to ``output[field_idx][sample_idx]``
        to match the expected batch format for DALI. Here, ``fields`` are the individual data
        fields of the sample data structure.

        Args:
            per_sample_res: List of sample results, where each sample is a list of field values.

        Returns:
            List of field results, where each field is a list of sample values.
        '''
        num_samples = len(per_sample_res)
        num_fields = len(per_sample_res[0])

        res = [None] * num_fields
        for f in range(num_fields):
            res[f] = [per_sample_res[s][f] for s in range(num_samples)]

        return res
