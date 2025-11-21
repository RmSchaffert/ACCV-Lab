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
import numpy as np

from accvlab.dali_pipeline_framework.inputs.sequence_sampler import SequenceSampler

# --------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------


def _check_sequence_integrity(slot_samples, sequence_start_to_length, slot_idx):
    """Helper function to check sequence integrity using the sequence mapping approach.

    Args:
        slot_samples: List of sample indices for a specific slot
        sequence_start_to_length: Mapping from sequence start index to sequence length
        slot_idx: Index of the slot being checked (for error messages)

    Raises:
        AssertionError: If sequence integrity is violated
    """
    curr_sequence_length = 0
    expected_sequence_length = -1  # Will be set to the length of the current sequence

    for i in range(len(slot_samples)):
        if curr_sequence_length > 0:
            assert slot_samples[i] == slot_samples[i - 1] + 1, (
                f"Slot {slot_idx}: "
                f"Non-consecutive index at position {i} in slot {slot_idx}: "
                f"previous sample {slot_samples[i-1]} is not the last sample of its sequence "
                f"(expected {slot_samples[i-1] + 1})"
            )
        else:
            assert slot_samples[i] in sequence_start_to_length, (
                f"Slot {slot_idx}: "
                f"Sample {slot_samples[i]} is not a valid sequence start index. "
                f"Sequence start indices: {list(sequence_start_to_length.keys())}. "
                f"{f'Last sample: {slot_samples[i-1]}' if i > 0 else 'This is the first sample'}"
            )
            expected_sequence_length = sequence_start_to_length[slot_samples[i]]

        curr_sequence_length += 1
        if curr_sequence_length == expected_sequence_length:
            curr_sequence_length = 0


def _get_sequence_id(sample_idx, sequence_lengths):
    """Helper function to map a sample index back to its sequence ID."""
    cumulative_length = 0
    for seq_id, length in enumerate(sequence_lengths):
        if sample_idx < cumulative_length + length:
            return seq_id
        cumulative_length += length
    raise ValueError(f"Sample index {sample_idx} out of range")


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_sequence_sampler_non_randomized():
    """Test sequence sampler without randomization - sequences should appear in order."""
    # Create sampler with 2 sample slots and 4 sequences of different lengths
    batch_size = 2
    sequence_lengths = [3, 4, 2, 3]  # Total: 12 samples
    seed = 1
    sampler = SequenceSampler(batch_size, sequence_lengths, seed, randomize=False)

    # Expected sequence distribution:
    # Slot 0: sequences [0, 2] -> samples [0,1,2, 7,8]
    # Slot 1: sequences [1, 3] -> samples [3,4,5,6, 9,10,11]

    # Get first batch
    batch1 = sampler.get_next_batch_indices()
    assert len(batch1) == batch_size

    # Slot 0 should start with sequence 0 (samples 0,1,2)
    assert batch1[0] == 0  # First sample of sequence 0

    # Slot 1 should start with sequence 1 (samples 3,4,5,6)
    assert batch1[1] == 3  # First sample of sequence 1

    # Get next batch - should continue with consecutive samples
    batch2 = sampler.get_next_batch_indices()
    assert batch2[0] == 1  # Second sample of sequence 0
    assert batch2[1] == 4  # Second sample of sequence 1

    batch3 = sampler.get_next_batch_indices()
    assert batch3[0] == 2  # Third sample of sequence 0
    assert batch3[1] == 5  # Third sample of sequence 1

    # Slot 0 should now start sequence 2 (samples 6,7)
    batch4 = sampler.get_next_batch_indices()
    assert batch4[0] == 7  # First sample of sequence 2
    assert batch4[1] == 6  # Fourth sample of sequence 1

    # Slot 1 should now start sequence 3 (samples 9,10,11)
    batch5 = sampler.get_next_batch_indices()
    assert batch5[0] == 8  # Second sample of sequence 2
    assert batch5[1] == 9  # First sample of sequence 3


def test_sequence_sampler_randomized_sequence_integrity():
    """Test that randomized sampling maintains sequence integrity - once a sequence starts,
    all samples from that sequence appear in consecutive batches at the same slot."""
    batch_size = 2
    sequence_lengths = [3, 4, 2, 3]  # Total: 12 samples
    sequence_start_to_length = {
        0: 3,
        3: 4,
        7: 2,
        9: 3,
    }  # Sequence start index to length mapping (for testing)
    for i in range(100):
        sampler = SequenceSampler(batch_size, sequence_lengths, seed=i, randomize=True)

        # Get multiple batches to observe sequence behavior
        batches = [sampler.get_next_batch_indices() for _ in range(10)]

        # Check that each slot maintains sequence integrity using the helper function
        for slot_idx in range(batch_size):
            slot_samples = [batch[slot_idx] for batch in batches]
            _check_sequence_integrity(slot_samples, sequence_start_to_length, slot_idx)


def test_sequence_sampler_randomized_distribution():
    """Test that randomized sampling distributes sequences across slots correctly.
    Each slot should iterate through a subset of sequences, and sequences should be
    distributed among slots so that each slot gets different sequences."""
    batch_size = 2
    sequence_lengths = [3, 4, 4, 3]  # 4 sequences

    for i in range(1000):

        sampler = SequenceSampler(batch_size, sequence_lengths, seed=i + 1000, randomize=True)

        # Stay within a single cycle and check that each slot gets 2 sequences and no sequence is repeated
        batches = [sampler.get_next_batch_indices() for _ in range(6)]

        # Check that each slot gets 2 different sequences
        slot0_sequences = set()
        slot1_sequences = set()

        for batch in batches:
            # Map sample indices back to sequence IDs
            slot0_seq = _get_sequence_id(batch[0], sequence_lengths)
            slot1_seq = _get_sequence_id(batch[1], sequence_lengths)

            slot0_sequences.add(slot0_seq)
            slot1_sequences.add(slot1_seq)

        both_slots_sequences = slot0_sequences.union(slot1_sequences)
        # Each slot should have seen multiple sequences
        assert len(slot0_sequences) == 2, f"Slot 0 saw sequences: {slot0_sequences}; expected 2 sequences"
        assert len(slot1_sequences) == 2, f"Slot 1 saw sequences: {slot1_sequences}; expected 2 sequences"
        assert (
            len(both_slots_sequences) == 4
        ), f"Both slots saw sequences: {both_slots_sequences}; expected 4 sequences"


def test_sequence_sampler_unequal_sequence_lengths():
    """Test behavior with sequences of different lengths, which may cause
    different slots to start new cycles at different times."""
    batch_size = 2
    sequence_lengths = [2, 5, 3, 4]  # Unequal lengths: 2, 5, 3, 4
    seed = 42
    sampler = SequenceSampler(batch_size, sequence_lengths, seed, randomize=True)

    # Get batches until we see a new cycle (when sequences repeat)
    batches = []
    seen_combinations = set()
    max_batches = 50  # Prevent infinite loop

    # Check that we can perform enough iterations to start new cycles.
    # Note that there are no assertions here, we just want to make sure there are no errors.
    for _ in range(max_batches):
        batch = sampler.get_next_batch_indices()
        batches.append(batch)

        # Create a signature for this batch to detect cycles
        batch_signature = tuple(sorted(batch))
        if batch_signature in seen_combinations:
            break  # We've seen this combination before, indicating a cycle
        seen_combinations.add(batch_signature)


def test_sequence_sampler_large_batch_size():
    """Test with larger batch size to ensure proper distribution across many slots."""
    batch_size = 4
    sequence_lengths = [3, 4, 2, 3, 5, 2]  # 6 sequences

    for i in range(1000):
        sampler = SequenceSampler(batch_size, sequence_lengths, seed=i + 2000, randomize=True)

        # Create mapping from sequence start index to sequence length
        sequence_start_to_length = {}
        cumulative_length = 0
        for seq_id, length in enumerate(sequence_lengths):
            sequence_start_to_length[cumulative_length] = length
            cumulative_length += length

        # Get several batches
        batches = []
        for _ in range(50):
            batches.append(sampler.get_next_batch_indices())

        # Check that each slot gets different sequences
        slot_sequences = [set() for _ in range(batch_size)]

        for batch in batches:
            for slot_idx in range(batch_size):
                seq_id = _get_sequence_id(batch[slot_idx], sequence_lengths)
                slot_sequences[slot_idx].add(seq_id)

        # Each slot should have seen multiple sequences
        for slot_idx in range(batch_size):
            assert (
                len(slot_sequences[slot_idx]) > 1
            ), f"Slot {slot_idx} only saw sequences: {slot_sequences[slot_idx]}"

        # Verify sequence integrity across all slots
        for slot_idx in range(batch_size):
            slot_samples = [batch[slot_idx] for batch in batches]
            _check_sequence_integrity(slot_samples, sequence_start_to_length, slot_idx)


def test_sequence_sampler_deterministic_with_seed():
    """Test that the same seed produces the same sequence of batches."""
    batch_size = 2
    sequence_lengths = [3, 4, 2, 3]
    seed = 123

    # Create two samplers with the same seed
    sampler1 = SequenceSampler(batch_size, sequence_lengths, seed, randomize=True)
    sampler2 = SequenceSampler(batch_size, sequence_lengths, seed, randomize=True)

    # Get several batches from both samplers
    batches1 = []
    batches2 = []
    for _ in range(50):
        batches1.append(sampler1.get_next_batch_indices())
        batches2.append(sampler2.get_next_batch_indices())

    # All batches should be identical
    for batch1, batch2 in zip(batches1, batches2):
        assert batch1 == batch2, f"Batches differ: {batch1} vs {batch2}"


def test_sequence_sampler_properties():
    """Test the properties and methods of the sequence sampler."""
    batch_size = 2
    sequence_lengths = [3, 4, 2, 3]
    seed = 42
    sampler = SequenceSampler(batch_size, sequence_lengths, seed, randomize=True)

    # Test properties
    assert sampler.length is None  # SequenceSampler doesn't support length
    assert sampler.is_epoch_based is False  # Not epoch-based

    # Test that reset raises an error
    with pytest.raises(RuntimeError, match="SequencesSampler is not epoch-based"):
        sampler.reset()


def test_sequence_sampler_edge_cases():
    """Test edge cases like single sequence, very short sequences."""
    # Single sequence
    with pytest.raises(
        AssertionError, match="The number of sequences must be at least the total batch size."
    ):
        SequenceSampler(2, [1], 42, randomize=False)

    # Very short sequences
    sampler = SequenceSampler(3, [1, 1, 1], 42, randomize=True)
    # Test that the batch has the correct length
    batch = sampler.get_next_batch_indices()
    assert len(batch) == 3

    # Test that multiple batches can be sampled without errors.
    for _ in range(100):
        batch = sampler.get_next_batch_indices()


if __name__ == "__main__":
    pytest.main([__file__])
