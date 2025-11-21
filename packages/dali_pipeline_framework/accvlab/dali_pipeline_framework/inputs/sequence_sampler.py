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

from typing import Sequence

import numpy as np

from .sampler_base import SamplerBase

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class SequenceSampler(SamplerBase):
    '''Sampler used to get consecutive samples from sequences contained in the dataset.

    For subsequent batches :math:`B_t` and :math:`B_{t+1}`, the individual samples in the batches with the
    same index :math:`i`, i.e. :math:`B_t[i]` and :math:`B_{t+1}[i]`, are subsequent samples inside a sequence
    :math:`S_j`, i.e. :math:`B_t[i] = S_j[k]` and :math:`B_{t+1}[i] = S_j[k+1]` (where :math:`j` is the index
    of the sequence in the dataset and :math:`k` is the index
    of the sample in the sequence :math:`S_j`), except when one sequence ends and another one begins.

    .. image:: ../images/sequence_sampling.png
        :align: center
        :alt: Sequence sampling illustration
        :width: 550px

    The sampling is performed by assigning for each "sample index slot" :math:`i` a set of sequences and then
    iterating through the sequences and outputting one sample at a time at the position :math:`i`.
    For this, the sequences are shuffled (represented by :math:`R_c` in the illustration) whenever a new cycle
    :math:`c` is started for one of the slots (:math:`R_0` and :math:`R_1` in the illustration correspond to
    the first two cycles).

    Note that each slot may reach a new cycle at different iterations :math:`t` as the total number of samples
    may vary for the individual slots. However, for each cycle :math:`c`, consistent shuffled lists
    :math:`R_c` are used for all slots (using consistent seeds for the shuffling).

    As the individual slots :math:`B_t[i]` may be in different cycles for a given iteration :math:`t`:

      - The cycles do not exactly correspond to epochs (as the cycle border is different for each slot).
        Therefore, this sampler is not epoch-based.
      - Although consistent shuffling is used to obtain :math:`R_c` across slots, the same sequence may still
        appear in multiple slots at the same time if the slots are in different cycles for a given iteration
        :math:`t` due to variable sequence length.
    '''

    def __init__(self, total_batch_size: int, sequence_lenghts: Sequence[int], seed: int, randomize=True):
        '''

        Args:
            total_batch_size: Total batch size (i.e. the combined batch size over all shards if sharding is
                used).
            sequence_lenghts: The lengths of the individual sequences. Note that the indices of the samples in
                the dataset must match the order of sequence lengths given, i.e. if the sequence lengths
                ``[10, 12]`` are given, then it is understood that the dataset contains 2 sequences,
                with the first containing the samples with indices in the range :math:`[0; 9]` and the
                second containing the samples with indices in the range :math:`[10; 21]`.
            seed: Random seed for shuffling sequences.
            randomize: Whether to shuffle sequences. If ``False``, sequences are used in order.

        '''
        assert (
            len(sequence_lenghts) >= total_batch_size
        ), "The number of sequences must be at least the total batch size."
        self._total_batch_size = total_batch_size
        self._sequence_lengths = sequence_lenghts
        self._sequence_to_global_index_mapping = self._get_sequence_to_global_index_mapping(sequence_lenghts)
        self._seed = seed
        self._randomize = randomize

        # Do not create the generators on creation to allow for pickling the object before
        # `get_next_batch_indices()` is called for the first time. This is important as this happens when the
        # worker processes which use this implementation are started.
        self._per_sample_slot_generators = [None] * total_batch_size
        self._generators_set = False

    @property
    @override
    def length(self):
        '''Length (size of a single epoch) is not defined as there are no clear epoch boundaries.

        Indicate this by returning ``None``.

        Returns:
            ``None``
        '''
        return None

    @override
    def get_next_batch_indices(self):
        '''Get the indices for the next batch of samples.

        Returns:
            List of sample indices for the next batch.
        '''
        if not self._generators_set:
            for i in range(self._total_batch_size):
                self._per_sample_slot_generators[i] = self._generate_for_sample_slot(i)
            self._generators_set = True
        res = [next(gen) for gen in self._per_sample_slot_generators]
        return res

    @property
    @override
    def is_epoch_based(self):
        '''Indicate that the sampler is not epoch-based by returning ``False``.

        Returns:
            ``False``
        '''
        return False

    @override
    def reset(self):
        '''Reset the sampler.

        Note that this method is not supported as the sampler is not epoch-based. Calling it will raise an
        error.

        Raises:
            RuntimeError: Will be raised if the method is called as the sampler is not epoch-based.
        '''
        raise RuntimeError("SequencesSampler is not epoch-based. the method `reset()` should not be called.")

    @staticmethod
    def _get_sequence_to_global_index_mapping(sequence_lengths: Sequence[int]) -> Sequence[Sequence[int]]:
        c = 0
        num_sequences = len(sequence_lengths)
        res = [None] * num_sequences
        for s in range(num_sequences):
            sl = sequence_lengths[s]
            res[s] = list(range(c, c + sl))
            c += sl
        return res

    def _generate_for_sample_slot(self, sample_slot_idx):
        '''Generate sample indices for a specific sample slot.

        Each slot is assigned a subset of sequences and cycles through them.
        When all sequences are exhausted for one slot, a new cycle begins with a new
        shuffled assignment of sequences. Note that the cycles are consistent for each slot.
        This means that while new cycles are started at different points in time for
        different slots, there is a fixed distribution of sequences across the slots for
        each cycle.

        Args:
            sample_slot_idx: Index of the sample slot (0 to total_batch_size-1).

        Yields:
            Sample indices for this slot.
        '''
        rand = np.random.default_rng(seed=self._seed)
        num_sequences = len(self._sequence_lengths)

        while True:
            if self._randomize:
                seq_of_seq = rand.permutation(num_sequences)
            else:
                seq_of_seq = np.array(range(num_sequences))
            seq_of_seq_to_use = seq_of_seq[sample_slot_idx :: self._total_batch_size]
            assert len(seq_of_seq_to_use) > 0, (
                f"The number of sequences to use is 0 for sample " f"slot {sample_slot_idx}."
            )
            for seq_id in seq_of_seq_to_use:
                indices_global_to_use = self._sequence_to_global_index_mapping[seq_id]
                assert len(indices_global_to_use) > 0, (
                    f"The number of indices to use is 0 for sample "
                    f"slot {sample_slot_idx} and sequence {seq_id}."
                )
                for idx in indices_global_to_use:
                    yield idx
