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

import os
import random
import logging
from typing import Dict
from torch.utils.data.sampler import Sampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoClipSamplerStream(Sampler):
    """
    Custom PyTorch Sampler for organizing video clips into batches.

    This sampler ensures that each batch contains samples from different clips,
    which is important for training stability and diversity.
    """

    def __init__(
        self,
        index_frame: Dict[str, Dict[str, int]],
        group_num: int,
        rank: int = 0,
        world_size: int = 1,
        use_cache: bool = True,
    ):
        """
        Initialize the VideoClipSampler.

        Args:
            index_frame: Dictionary mapping video directories to clip frame counts.
            group_num: Number of clips to process in each group.
            rank: Process rank for distributed training.
            world_size: Total number of processes for distributed training.
        """
        self.index_frame = index_frame
        self.group_num = group_num
        self.rank = rank
        self.world_size = world_size
        self.use_cache = use_cache

        # Validate inputs
        if group_num <= 0:
            raise ValueError(f"group_num must be positive, got {group_num}")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")

        self._generate_groups()
        logger.info(f"Initialized sampler with {len(self.batches)} batches")

    def _generate_groups(self):
        """Generate groups of video clips for batch processing."""
        # Collect all (clip_path, frame_idx) pairs
        self.groups_lst = []
        for video_dir, clip_info in self.index_frame.items():
            for clip_id, frame_count in clip_info.items():
                clip_path = os.path.join(video_dir, clip_id)
                for i in range(frame_count):
                    self.groups_lst.append((clip_path, i, self.use_cache))

        # Organize into groups
        self.groups = []
        self.group_size = len(self.groups_lst) // self.group_num
        for i in range(0, self.group_size * self.group_num, self.group_size):
            group = self.groups_lst[i : i + self.group_size]
            self.groups.append(group)

        # Create batches (each batch contains one sample from each group)
        self.batches = []
        for i in range(self.group_size):
            batch = []
            for group in self.groups:
                if i < len(group):
                    batch.append(group[i])
            self.batches.append(batch)

        self.total_batches = len(self.batches)

    def __iter__(self):
        """
        Iterate over batches of samples for distributed training.

        Yields:
            Batches of (clip_path, frame_idx) tuples.
        """
        for batch_idx in range(self.rank, self.total_batches, self.world_size):
            yield self.batches[batch_idx]

    def __len__(self) -> int:
        """
        Get the number of batches for this process.

        Returns:
            Number of batches for the current rank.
        """
        return (self.total_batches + self.world_size - 1 - self.rank) // self.world_size


class VideoClipSamplerRandom(Sampler):
    """
    Custom PyTorch Sampler for random video clip sampling.

    This sampler randomly selects clips and frames for each iteration,
    providing more diverse training data compared to sequential sampling.
    """

    def __init__(
        self, index_frame: Dict[str, Dict[str, int]], group_num: int, rank: int = 0, world_size: int = 1
    ):
        """
        Initialize the VideoClipSamplerRandom.

        Args:
            index_frame: Dictionary mapping video directories to clip frame counts.
            group_num: Number of clips to process in each batch.
            rank: Process rank for distributed training.
            world_size: Total number of processes for distributed training.
        """
        self.index_frame = index_frame
        self.group_num = group_num
        self.rank = rank
        self.world_size = world_size

        # Validate inputs
        if group_num <= 0:
            raise ValueError(f"group_num must be positive, got {group_num}")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")

        self._generate_clip_list()
        logger.info(f"Initialized random sampler with {len(self.clip_list)} clips")

    def _generate_clip_list(self):
        """Generate list of available clips with their frame counts."""
        self.clip_list = []
        for video_dir, clip_info in self.index_frame.items():
            for clip_id, frame_count in clip_info.items():
                clip_path = os.path.join(video_dir, clip_id)
                self.clip_list.append((clip_path, frame_count))

        logger.debug(f"Generated {len(self.clip_list)} clips for random sampling")

    def __iter__(self):
        """
        Iterate over randomly selected batches of samples.

        Yields:
            Batches of (clip_path, frame_idx, use_cache) tuples where use_cache is always False.
        """
        # Set random seed for reproducibility in distributed training
        random.seed(42 + self.rank)

        for i in range(1000):
            batch = []
            # Randomly select group_num different clips
            selected_clips = random.sample(self.clip_list, min(self.group_num, len(self.clip_list)))

            for clip_path, frame_count in selected_clips:
                # Randomly select a frame index for this clip
                frame_idx = random.randint(0, frame_count - 1)

                # Always use cache=False for random sampling
                batch.append((clip_path, frame_idx, False))

            # If we have fewer clips than group_num, fill with random selections
            while len(batch) < self.group_num:
                clip_path, frame_count = random.choice(self.clip_list)
                frame_idx = random.randint(0, frame_count - 1)
                batch.append((clip_path, frame_idx, False))

            yield batch

    def __len__(self) -> int:
        """
        Get the number of batches for this process.

        Returns:
            Number of batches for the current rank (infinite for random sampling).
        """
        # For random sampling, we return a large number to allow infinite iteration
        # The actual iteration is controlled by the DataLoader's iteration logic
        return 1000  # Large number to allow many iterations
