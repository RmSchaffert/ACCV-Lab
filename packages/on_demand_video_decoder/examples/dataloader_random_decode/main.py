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


"""
``accvlab.on_demand_video_decoder`` - DataLoader Random Decode Example

This example demonstrates how to use ``accvlab.on_demand_video_decoder`` library with PyTorch DataLoader
for efficient random-access video decoding in distributed training scenarios. The example shows
how to create a custom Dataset and Sampler that work with PyTorch's DataLoader to provide
GPU-accelerated video decoding for machine learning training pipelines.

This example uses CreateGopDecoder for random access patterns.

Key Features Demonstrated:
- Custom PyTorch Dataset with lazy initialization of GPU decoder (CreateGopDecoder)
- Random frame access capability
- Custom Sampler for distributed training support
- Multi-camera video clip processing
- Batch processing with configurable group sizes
- Performance profiling with NVTX markers
- Distributed training compatibility

This example is particularly useful for:
- Multi-camera autonomous driving datasets (e.g., nuScenes)
- Video understanding tasks requiring multiple synchronized video streams
- Large-scale distributed training with video data
- Random sampling video processing pipelines
"""

import os
import time
import argparse
import json
import sys
from typing import Dict, List, Tuple, Any, Optional

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))
import dataloader_separation_decode.video_clip_sampler as video_clip_sampler

import accvlab.on_demand_video_decoder as nvc

# Configuration constants
NUM_CAMERAS = 6
DEFAULT_INDEX_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index_frame.json")
DEFAULT_GROUP_NUM = 4
DEFAULT_NUM_WORKERS = 2
WARMUP_ITERATIONS = 10
PROGRESS_REPORT_INTERVAL = 10


def load_index_frame(json_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load frame index information from a JSON file.

    The JSON file should have the following structure:
    {
        "video_dir1": {
            "clip0id": frame_count,
            "clip1id": frame_count,
            ...
        },
        "video_dir2": {
            "clip0id": frame_count,
            "clip1id": frame_count,
            ...
        }
    }

    Args:
        json_file: Path to the JSON file containing frame index information

    Returns:
        Dictionary containing video directory to clip frame count mappings

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Index file not found: {json_file}")

    with open(json_file, 'r') as f:
        index_frame = json.load(f)
    return index_frame


def is_distributed() -> bool:
    """Check if we're in a distributed training environment."""
    return "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def setup_distributed() -> Tuple[int, int]:
    """
    Setup distributed training environment if running in distributed mode.

    Returns:
        Tuple of (local_rank, world_size)
    """
    if is_distributed():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        print(f"Distributed mode - Local rank: {local_rank}, World size: {world_size}")
        return local_rank, world_size
    else:
        print("Single process mode")
        return 0, 1


def cleanup_distributed() -> None:
    """Cleanup distributed training resources if initialized."""
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


# .. doc-marker-begin: dataset-random
class VideoClipDataset(Dataset):
    """
    PyTorch Dataset for video clip decoding using ``accvlab.on_demand_video_decoder``.

    This dataset provides GPU-accelerated video decoding capabilities with lazy
    initialization of the decoder to optimize memory usage. It supports multi-camera
    video clips and is designed for distributed training scenarios.

    This implementation uses CreateGopDecoder for random access.

    Attributes:
        device_id: GPU device ID for decoder initialization
        index_frame: Frame index information loaded from JSON
        num_cameras: Number of cameras per clip
        group_num: Number of clips to process in each batch
        use_fast_init: Whether to use fast initialization with GetFastInitInfo
        _is_initialized: Flag indicating if decoder has been initialized
        _nvc_decoder: NVIDIA video decoder instance (CreateGopDecoder)
        fast_stream_infos: Fast initialization info for decoder
    """

    def __init__(
        self,
        index_frame: Dict[str, Dict[str, int]],
        group_num: int,
        device_id: int,
        num_cameras: int = NUM_CAMERAS,
        use_fast_init: bool = False,
    ):
        """
        Initialize the VideoClipDataset for random access.

        Args:
            index_frame: Frame index information from JSON file
            group_num: Number of clips to process in each batch
            device_id: GPU device ID for decoder initialization
            num_cameras: Number of cameras per clip (default: 6)
            use_fast_init: Whether to use fast initialization (default: False)
        """
        self.device_id = device_id
        self.index_frame = index_frame
        self.num_cameras = num_cameras
        self.group_num = group_num
        self._is_initialized = False
        self._nvc_decoder = None
        self._use_fast_init = use_fast_init
        self.fast_stream_infos = None

    def __len__(self) -> int:
        """
        Get the total number of batches in the dataset.

        Returns:
            Total number of batches (total frames // group_num)
        """
        total_frames = 0
        for video_dir, clip_info in self.index_frame.items():
            for clip_id, frame_count in clip_info.items():
                total_frames += frame_count

        return total_frames // self.group_num

    def _lazy_init(self, index) -> None:
        """
        Lazy initialize the NVIDIA video decoder (CreateGopDecoder).

        This method initializes the decoder only when first needed, which helps
        optimize memory usage and startup time.
        """
        if self._is_initialized:
            return

        try:
            self._nvc_decoder = nvc.CreateGopDecoder(
                maxfiles=self.num_cameras,  # Maximum number of files to use
                iGpu=self.device_id,  # GPU ID
            )
            if self._use_fast_init:
                file_lists = [
                    os.path.join(index[0][0], f) for f in os.listdir(index[0][0]) if f.endswith('.mp4')
                ]
                self.fast_stream_infos = nvc.GetFastInitInfo(file_lists)

            self._is_initialized = True
            print(f"Random video decoder (CreateGopDecoder) initialized on GPU {self.device_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize random video decoder on GPU {self.device_id}: {e}")

    def __getitem__(self, index: List[Tuple[str, int]]) -> List[List[torch.Tensor]]:
        """
        Get a batch of video frames for the given index.

        Args:
            index: List of tuples containing (clip_path, frame_idx) pairs

        Returns:
            List of lists of decoded video frames as PyTorch tensors

        Raises:
            RuntimeError: If decoder initialization fails
            ValueError: If input index format is invalid
        """
        if not isinstance(index, list):
            raise ValueError(f"Expected list of tuples, got {type(index)}")

        self._lazy_init(index)
        decoded_batch = []

        for clip_path, frame_idx, _ in index:
            try:
                # Find all MP4 files in the clip directory
                video_paths = [
                    os.path.join(clip_path, f) for f in os.listdir(clip_path) if f.endswith('.mp4')
                ]

                if not video_paths:
                    raise ValueError(f"No MP4 files found in {clip_path}")

                # Create frame index list for all videos (same frame for synchronized cameras)
                frame_indices = [frame_idx] * len(video_paths)

                # Decode frames from all cameras using CreateGopDecoder
                if self._use_fast_init:
                    decoded_frames = self._nvc_decoder.DecodeN12ToRGB(
                        video_paths, frame_indices, True, self.fast_stream_infos
                    )
                else:
                    decoded_frames = self._nvc_decoder.DecodeN12ToRGB(video_paths, frame_indices)

                # Convert to PyTorch tensors and add batch dimension
                frame_tensors = [
                    torch.unsqueeze(torch.as_tensor(df, device=f'cuda:{self.device_id}'), 0)
                    for df in decoded_frames
                ]

                decoded_batch.append(frame_tensors)

            except Exception as e:
                raise RuntimeError(f"Failed to decode frames from {clip_path}: {e}")

        return decoded_batch


# .. doc-marker-end: dataset-random


def main():
    """
    Main function demonstrating random-access video clip decoding with PyTorch DataLoader.

    This function sets up distributed training, initializes the dataset and dataloader,
    and runs a performance benchmark with warmup iterations and detailed metrics.
    Uses CreateGopDecoder for random frame access.
    """
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Random video clip decoding example using ``accvlab.on_demand_video_decoder`` (CreateGopDecoder) with PyTorch DataLoader"
    )
    parser.add_argument(
        "--index_file", type=str, help='Path to the index_frame JSON file', default=DEFAULT_INDEX_FILE
    )
    parser.add_argument(
        "--group_num", type=int, default=DEFAULT_GROUP_NUM, help='Number of clips to process in each batch'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help='Number of worker processes for data loading',
    )
    parser.add_argument(
        "--use_fast_init",
        action='store_true',
        help='Use fast initialization with GetFastInitInfo',
    )
    args = parser.parse_args()

    print(f"Using index_file: {args.index_file}")
    print(f"Using group_num: {args.group_num}")
    print(f"Using num_workers: {args.num_workers}")
    print(f"Using use_fast_init: {args.use_fast_init}")

    # Setup distributed training if in distributed environment
    local_rank, local_world_size = setup_distributed()

    # .. doc-marker-begin: training-setup-random
    # Load frame index information
    index_frame = load_index_frame(args.index_file)
    print(f"Loaded index frame with {len(index_frame)} video directories")

    # Create dataset and dataloader for random access
    dataset = VideoClipDataset(
        index_frame=index_frame,
        group_num=args.group_num,
        device_id=local_rank,
        use_fast_init=args.use_fast_init,
    )

    # Use random sampler for random access
    sampler = video_clip_sampler.VideoClipSamplerRandom(
        index_frame=index_frame, group_num=args.group_num, rank=local_rank, world_size=local_world_size
    )

    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=args.num_workers, pin_memory=False
    )

    loader_iter = iter(dataloader)

    # Warmup phase
    print(f"\nStarting warmup phase ({WARMUP_ITERATIONS} iterations)...")
    for i in range(WARMUP_ITERATIONS):
        nvtx.range_push(f"warmup_batch_{i}")

        nvtx.range_push("next_batch")
        batch = next(loader_iter)
        nvtx.range_pop()  # next_batch

        nvtx.range_pop()  # warmup_batch_{i}
        print(f"Warmup {i + 1}/{WARMUP_ITERATIONS} completed")

    # Performance benchmark phase
    print(f"\nStarting performance benchmark...")
    started_at = time.perf_counter()
    current_time = 0
    frame_loaded = 0
    samples_loaded = 0
    batches_loaded = 0
    load_gaps = []
    load_started_at = time.perf_counter()

    i = 0
    while True:
        nvtx.range_push(f"batch_{i}")

        nvtx.range_push("next_batch")
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        nvtx.range_pop()  # next_batch

        # Process batch (example: print shape of first frame)
        # print(f"Batch {i}: Frame shape {batch[0][0].shape}")

        nvtx.range_push("load_batch")
        load_ended_at = time.perf_counter()
        load_gaps.append(load_ended_at - load_started_at)
        load_started_at = time.perf_counter()

        batches_loaded += 1
        samples_loaded += args.group_num
        frame_loaded += NUM_CAMERAS * args.group_num

        current_time = time.perf_counter()

        if i % PROGRESS_REPORT_INTERVAL == 0:
            elapsed_time = current_time - started_at
            throughput_fps = frame_loaded / elapsed_time if elapsed_time > 0 else 0
            throughput_samples = samples_loaded / elapsed_time if elapsed_time > 0 else 0

            print(f"Batch: {i} - {len(batch)} frames in {load_gaps[-1]:.4f} seconds")
            print(f"Elapsed time: {elapsed_time:.4f}s")
            print(f"Samples loaded: {samples_loaded}")
            print(f"Throughput: {throughput_fps:.2f} frames/second, {throughput_samples:.2f} samples/second")

        nvtx.range_pop()  # load_batch
        nvtx.range_pop()  # batch_{i}

        i += 1

    # Final performance report
    # ended_at = time.perf_counter()
    ended_at = current_time
    total_time = ended_at - started_at

    print(f"\n=== Performance Summary ===")
    print(f"Frames loaded: {frame_loaded}")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Throughput: {frame_loaded / total_time:.2f} frames per second")
    print(f"Throughput: {samples_loaded / total_time:.2f} samples per second")
    print(f"Throughput: {batches_loaded / total_time:.2f} batches per second")
    print(f"Number of workers: {args.num_workers}")

    if load_gaps:
        avg_load_time = sum(load_gaps) / len(load_gaps)
        print(f"Average batch load time: {avg_load_time:.4f} seconds")
    # .. doc-marker-end: training-setup-random

    cleanup_distributed()


if __name__ == "__main__":
    main()
