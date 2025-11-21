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

import numpy as np

from nvidia.dali import types

from ..pipeline import SampleDataGroup

from .callable_base import CallableBase
from .data_provider import DataProvider

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class ShuffledShardedInputCallable(CallableBase):
    '''Input callable supporting shuffling and sharding.

    This class implements data randomization by shuffling, as well as distributing the data into multiple
    shards. The shuffling and sharding is done following the general approach outlined in [1].

    The randomization can be disabled, in which case the data is read in sequential order.

    Note:
        If the data set is not divisible by the batch size (in case of sharding, the total batch size over all
        shards), the incomplete batch at the end of each epoch will be dropped.

    [1] https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html#Shuffling-and-sharding

    '''

    def __init__(
        self,
        data_provider: DataProvider,
        batch_size: int,
        shard_id: int = 0,
        num_shards: int = 1,
        shuffle: bool = False,
        seed: int = 21,
    ):
        '''

        Args:
            data_provider: Data provider (following the interface defined in :class:`DataProvider`) used to obtain the samples and additional data.
            batch_size: Batch size
            shard_id: Shard ID. Needs to be set if sharding is used.
            num_shards: Total number of shards. Needs to be set if sharding is used
            shuffle: Whether to shuffle the data
            seed: Seed used for the shuffling. If sharding is used, the input callables for all shards need to use the same seed.
        '''

        self._data_provider = data_provider

        self._batch_size = batch_size
        self._shard_id = shard_id
        self._num_shards = num_shards
        self._shuffle = shuffle
        self._seed = seed

        self._data_len = self._data_provider.get_number_of_samples()

        self._shard_size = self._data_len // self._num_shards
        self._shard_offset = self._shard_size * self._shard_id
        self._full_iterations = self._shard_size // self._batch_size

        self._permutation = None
        self._last_seen_epoch = -1

    @property
    @override
    def used_sample_data_structure(self) -> SampleDataGroup:
        '''Get the data format blueprint used for the individual samples'''
        res = self._data_provider.sample_data_structure
        res.set_apply_mapping(False)
        return res

    @override
    def __call__(self, sample_info: types.SampleInfo) -> tuple:

        if sample_info.idx_in_epoch >= self._shard_size:
            raise StopIteration

        # If this is a new epoch ...
        if self._last_seen_epoch != sample_info.epoch_idx:
            # ... set up a new permutation
            self._permutation = self._setup_permutation(sample_info.epoch_idx)
            # and indcate that this is not a new epoch enymore, preventing a new permutation
            # to be generated at the next call to this function
            self._last_seen_epoch = sample_info.epoch_idx

        index_in_shard = self._shard_offset + sample_info.idx_in_epoch % self._shard_size
        index_to_use = self._permutation[index_in_shard]

        # Note: Next line can be used to always load the same sample (e.g. for debugging)
        # index_to_use = 10

        sample_data = self._data_provider.get_data(index_to_use)

        return sample_data.get_data()

    @property
    @override
    def length(self) -> Optional[int]:
        '''Number of full iterations (complete batches) in the epoch.

        If the underlying sampler is not epoch-based, ``None`` is returned.
        '''
        return self._full_iterations

    def _setup_permutation(self, epoch_idx: int) -> np.ndarray:
        '''Set up the permutation for shuffling samples in a given epoch.

        Creates a permutation array that determines the order of samples for
        the specified epoch. If shuffling is disabled, returns a sequential
        permutation (0, 1, 2, ...).

        Args:
            epoch_idx: The epoch index to generate the permutation for.

        Returns:
            A numpy array containing the permutation indices for the epoch.
        '''
        # Define shuffling order if needed.
        if self._shuffle:
            # Note that the seed will be the same for multiple instances (for multi-GPU training) if self._seed is the same.
            # This has to be the case to prevent different sharding splits in different instances, which would lead to reuse of some samples and skipping of others.
            perm = np.random.default_rng(seed=self._seed + epoch_idx).permutation(self._data_len)
        else:
            perm = np.array(range(self._data_len))
        return perm


if __name__ == "__main__":
    import types as tps
    import cv2

    base_dir = '/home/rschaffert/Work/datasets/nuscenes/mini/'
    nuscenes_version = 'v1.0-mini'

    reader = NuScenesReader(base_dir, nuscenes_version)

    callable = NuScenesInputCallable(reader, 4, use_single_images=False)

    sample_info = tps.SimpleNamespace()

    sample_info.epoch_idx = 0
    sample_info.idx_in_epoch = 0

    raw_data = callable(sample_info)
    sample_data = callable.used_sample_data_structure.with_data_set(raw_data)

    print(sample_data)

    for i in range(6):
        cv2.imwrite(f"temp_out/image_{i}.png", sample_data[i]["image"])

        bbox_image = np.zeros(sample_data[0]["image"].shape[0:2])
        bboxes = sample_data[0]["annotation"]["bboxes"]
        categories = sample_data[0]["annotation"]["categories"]
        num_bboxes = bboxes.shape[0]

        for j in range(num_bboxes):
            bbox_image = cv2.rectangle(
                bbox_image,
                (bboxes[j, :2]).astype(int),
                (bboxes[j, 2:]).astype(int),
                color=np.random.randint(155, 255),
            )
        cv2.imwrite(f"temp_out/bbox_{i}.png", bbox_image)
