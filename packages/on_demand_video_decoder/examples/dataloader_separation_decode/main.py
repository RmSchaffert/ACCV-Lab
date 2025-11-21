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
DataLoader Separation Decode Example

This example demonstrates how to use ``accvlab.on_demand_video_decoder`` library with PyTorch DataLoader
for efficient separation-based video decoding in distributed training scenarios.

The example shows how to create a custom PyTorch Dataset and Sampler that work with PyTorch's
DataLoader to provide GPU-accelerated video decoding for machine learning training pipelines.
It's particularly useful for multi-camera datasets where you need to process synchronized
video streams from multiple cameras simultaneously.

Key Features:
- Custom PyTorch Dataset with lazy initialization of GPU decoder
- Custom Sampler with distributed training support
- Multi-camera video stream processing
- Performance profiling with NVTX markers
- Comprehensive error handling and validation
- Separation-based video decoding for efficient memory usage

Usage:
    python main.py --index_file /path/to/index_frame.json --group_num 4 --num_workers 2

    # For distributed training:
    python -m torch.distributed.run --nproc_per_node=2 main.py --index_file /path/to/index_frame.json --group_num 4 --num_workers 2
"""

import os
import time
import argparse
import json
import logging
import random
from typing import Dict, List, Tuple, Any
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler

import video_demuxing
import video_transforms
import video_clip_sampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method('fork', force=True)

# Constants
NUM_CAMERAS = 6
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_LOG_INTERVAL = 10


def load_index_frame(json_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load frame index information from JSON file.

    Args:
        json_file: Path to the JSON file containing frame index information.

    Returns:
        Dictionary containing video directory to clip frame count mapping.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    try:
        with open(json_file, 'r') as f:
            index_frame = json.load(f)
        logger.info(f"Successfully loaded index frame from {json_file}")
        return index_frame
    except FileNotFoundError:
        logger.error(f"Index file not found: {json_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_file}: {e}")
        raise


def is_distributed() -> bool:
    """Check if we're in a distributed environment."""
    return "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def setup_distributed() -> Tuple[int, int]:
    """
    Setup distributed training if in distributed environment.

    Returns:
        Tuple of (local_rank, world_size).
    """
    if is_distributed():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed mode - Local rank: {local_rank}, World size: {world_size}")
        return local_rank, world_size
    else:
        logger.info("Single process mode")
        return 0, 1


def cleanup_distributed():
    """Cleanup distributed training if initialized."""
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


# .. doc-marker-begin: dataset-separation
class VideoClipDatasetWithPyNVVideo(Dataset):
    """
    Custom PyTorch Dataset for video clip processing with accvlab.on_demand_video_decoder.

    This dataset provides lazy initialization of the GPU decoder and efficient
    video frame extraction for multi-camera datasets.
    """

    def __init__(
        self, index_frame: Dict[str, Dict[str, int]], group_num: int, num_cameras: int = NUM_CAMERAS
    ):
        """
        Initialize the VideoClipDatasetWithPyNVVideo.

        Args:
            index_frame: Dictionary mapping video directories to clip frame counts.
            group_num: Number of clips to process in each group.
            num_cameras: Number of cameras in the dataset.
        """
        self.index_frame = index_frame
        self.num_cameras = num_cameras
        self.group_num = group_num
        self._is_initialized = False
        self._indexing_demuxer = None

        # Validate inputs
        if group_num <= 0:
            raise ValueError(f"group_num must be positive, got {group_num}")
        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")

        logger.info(f"Initialized dataset with {num_cameras} cameras, group_num={group_num}")

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            Total number of samples.
        """
        total_frames = 0
        for video_dir, clip_info in self.index_frame.items():
            for clip_id, frame_count in clip_info.items():
                total_frames += frame_count

        length = total_frames // self.group_num
        logger.debug(f"Dataset length: {length} (total_frames={total_frames}, group_num={self.group_num})")
        return length

    def __lazy_init__(self, use_cache: bool):
        """Lazy initialize the video demuxer to optimize memory usage."""
        if self._is_initialized:
            return

        try:
            self._indexing_demuxer = video_demuxing.IndexingDemuxerOndemand(
                batch_size=self.group_num,
                num_cameras=self.num_cameras,
                use_cache=use_cache,
            )
            self._is_initialized = True
            logger.debug("Video demuxer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize video demuxer: {e}")
            raise

    def __getitem__(self, index: List[Tuple[str, int]]) -> List[Any]:
        """
        Get video frames for the given index.

        Args:
            index: List of (clip_path, frame_idx) tuples.

        Returns:
            List of episode buffers containing video frame data.

        Raises:
            ValueError: If index format is invalid.
            RuntimeError: If video demuxer fails to process frames.
        """
        if not isinstance(index, list):
            raise ValueError(f"Expected list of tuples, got {type(index)}")

        use_cache = index[0][2]
        self.__lazy_init__(use_cache)
        episode_buffers = []

        try:
            for i, (clip_path, frame_idx, _) in enumerate(index):
                # Find all MP4 files in the clip directory
                video_paths = [
                    os.path.join(clip_path, f) for f in os.listdir(clip_path) if f.endswith('.mp4')
                ]

                if not video_paths:
                    logger.warning(f"No MP4 files found in {clip_path}")
                    continue

                # Create frame indices for all cameras
                frame_indices = [frame_idx] * len(video_paths)

                # Update demuxer paths and get packet buffers
                self._indexing_demuxer.update_path(video_paths)
                episode_buffers.append(
                    self._indexing_demuxer.packet_buffers_for_frame_idx_list(frame_indices, sample_idx=i)
                )

            return episode_buffers

        except Exception as e:
            logger.error(f"Error processing index {index}: {e}")
            raise RuntimeError(f"Failed to process video frames: {e}")


# .. doc-marker-end: dataset-separation


def run_warmup(
    dataloader: DataLoader,
    decode_video_on_demand: video_transforms.DecodeVideoOnDemand,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
) -> None:
    """
    Run warmup iterations to initialize GPU kernels and caches.

    Args:
        dataloader: PyTorch DataLoader instance.
        decode_video_on_demand: Video decoder instance.
        warmup_iterations: Number of warmup iterations.
    """
    logger.info(f"Starting warmup with {warmup_iterations} iterations")
    loader_iter = iter(dataloader)

    for i in range(warmup_iterations):
        try:
            nvtx.range_push(f"warmup_batch_{i}")

            nvtx.range_push("warmup_next_batch")
            batch = next(loader_iter)
            nvtx.range_pop()

            nvtx.range_push("warmup_decode_batch")
            decode_video_on_demand.transform(batch)
            nvtx.range_pop()

            nvtx.range_pop()
            logger.debug(f"Warmup iteration {i + 1}/{warmup_iterations} completed")

        except StopIteration:
            logger.warning(f"DataLoader exhausted during warmup at iteration {i}")
            break
        except Exception as e:
            logger.error(f"Error during warmup iteration {i}: {e}")
            continue

    logger.info("Warmup completed")


def run_benchmark(
    dataloader: DataLoader,
    decode_video_on_demand: video_transforms.DecodeVideoOnDemand,
    group_num: int,
    log_interval: int = DEFAULT_LOG_INTERVAL,
) -> Dict[str, float]:
    """
    Run the main benchmark loop.

    Args:
        dataloader: PyTorch DataLoader instance.
        decode_video_on_demand: Video decoder instance.
        group_num: Number of clips per batch.
        log_interval: Interval for logging progress.

    Returns:
        Dictionary containing performance metrics.
    """
    logger.info("Starting benchmark")
    loader_iter = iter(dataloader)

    started_at = time.perf_counter()
    frame_loaded = 0
    samples_loaded = 0
    batches_loaded = 0
    load_gaps = []
    load_started_at = time.perf_counter()
    current_time = 0
    i = 0

    while True:
        nvtx.range_push(f"batch_{i}")

        # Get next batch
        nvtx.range_push("next_batch")
        try:
            batch = next(loader_iter)
        except StopIteration:
            logger.info("DataLoader exhausted, benchmark complete")
            break
        nvtx.range_pop()

        # Decode batch
        nvtx.range_push("decode_batch")
        try:
            res = decode_video_on_demand.transform(batch)
        except Exception as e:
            logger.error(f"Error decoding batch {i}: {e}")
            continue
        nvtx.range_pop()

        # Record timing and statistics
        nvtx.range_push("load_batch")
        load_ended_at = time.perf_counter()
        load_gaps.append(load_ended_at - load_started_at)
        load_started_at = time.perf_counter()

        batches_loaded += 1
        samples_loaded += group_num
        frame_loaded += NUM_CAMERAS * group_num

        current_time = time.perf_counter()

        # Log progress
        if i % log_interval == 0:
            elapsed_time = current_time - started_at
            throughput_fps = frame_loaded / elapsed_time if elapsed_time > 0 else 0
            throughput_sps = samples_loaded / elapsed_time if elapsed_time > 0 else 0

            logger.info(f"Batch: {i} - {len(batch)} frames in {load_gaps[-1]:.4f} seconds")
            logger.info(f"Current Time taken: {elapsed_time:.4f}s")
            logger.info(f"Current Samples loaded: {samples_loaded}")
            logger.info(f"Current Throughput: {throughput_fps:.2f} frames per second")

        nvtx.range_pop()  # load_batch
        nvtx.range_pop()  # batch_{i}

        i += 1

    # Calculate final metrics
    # ended_at = time.perf_counter()
    ended_at = current_time
    total_time = ended_at - started_at

    metrics = {
        'frames_loaded': frame_loaded,
        'samples_loaded': samples_loaded,
        'batches_loaded': batches_loaded,
        'total_time': total_time,
        'throughput_fps': frame_loaded / total_time if total_time > 0 else 0,
        'throughput_sps': samples_loaded / total_time if total_time > 0 else 0,
        'throughput_bps': batches_loaded / total_time if total_time > 0 else 0,
        'avg_batch_time': sum(load_gaps) / len(load_gaps) if load_gaps else 0,
    }

    return metrics


def print_performance_summary(metrics: Dict[str, float], num_workers: int) -> None:
    """
    Print a formatted performance summary.

    Args:
        metrics: Dictionary containing performance metrics.
        num_workers: Number of worker processes used.
    """
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Frames loaded: {metrics['frames_loaded']:,}")
    print(f"Samples loaded: {metrics['samples_loaded']:,}")
    print(f"Batches loaded: {metrics['batches_loaded']:,}")
    print(f"Time taken: {metrics['total_time']:.2f} seconds")
    print(f"Throughput: {metrics['throughput_fps']:.2f} frames per second")
    print(f"Throughput: {metrics['throughput_sps']:.2f} samples per second")
    print(f"Throughput: {metrics['throughput_bps']:.2f} batches per second")
    print(f"Average batch load time: {metrics['avg_batch_time']:.4f} seconds")
    print(f"Number of workers: {num_workers}")
    print("=" * 50)


def main():
    """Main function to run the video decoding benchmark."""
    parser = argparse.ArgumentParser(
        description="``accvlab.on_demand_video_decoder`` DataLoader Separation Decode Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "index_frame.json"),
        help='Path to the index_frame JSON file',
    )
    parser.add_argument("--group_num", type=int, default=4, help='Number of clips to process in each batch')
    parser.add_argument(
        "--num_workers", type=int, default=2, help='Number of worker processes for data loading'
    )
    parser.add_argument(
        "--disable_cache", action="store_true", help='if True, use cache for video demuxer packets buffer'
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS, help='Number of warmup iterations'
    )
    parser.add_argument(
        "--log_interval", type=int, default=DEFAULT_LOG_INTERVAL, help='Interval for logging progress'
    )
    parser.add_argument(
        "--frame_read_type",
        type=str,
        choices=['stream', 'random'],
        default='stream',
        help='Frame reading type: stream (sequential) or random (random sampling)',
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.index_file):
        logger.error(f"Index file not found: {args.index_file}")
        return 1

    if args.group_num <= 0:
        logger.error("group_num must be positive")
        return 1

    if args.num_workers < 0:
        logger.error("num_workers must be non-negative")
        return 1

    logger.info(f"Using index_file: {args.index_file}")
    logger.info(f"Using group_num: {args.group_num}")
    logger.info(f"Using num_workers: {args.num_workers}")

    try:
        # Setup distributed training
        # .. doc-marker-begin: training-setup-separation
        local_rank, local_world_size = setup_distributed()

        # Load index frame
        index_frame = load_index_frame(args.index_file)

        # Create dataset and dataloader
        dataset = VideoClipDatasetWithPyNVVideo(index_frame=index_frame, group_num=args.group_num)

        # Create sampler based on frame_read_type
        if args.frame_read_type == 'stream':
            sampler = video_clip_sampler.VideoClipSamplerStream(
                index_frame=index_frame,
                group_num=args.group_num,
                rank=local_rank,
                world_size=local_world_size,
                use_cache=not args.disable_cache,
            )
        elif args.frame_read_type == 'random':
            sampler = video_clip_sampler.VideoClipSamplerRandom(
                index_frame=index_frame,
                group_num=args.group_num,
                rank=local_rank,
                world_size=local_world_size,
            )
        else:
            raise ValueError(f"Unknown frame_read_type: {args.frame_read_type}")

        dataloader = DataLoader(
            dataset, batch_size=1, sampler=sampler, num_workers=args.num_workers, pin_memory=False
        )

        # Create video decoder
        decode_video_on_demand = video_transforms.DecodeVideoOnDemand(
            device_id=local_rank, num_cameras=NUM_CAMERAS
        )

        # Run warmup
        run_warmup(dataloader, decode_video_on_demand, args.warmup_iterations)

        # Run benchmark
        metrics = run_benchmark(dataloader, decode_video_on_demand, args.group_num, args.log_interval)

        # Print results
        print_performance_summary(metrics, args.num_workers)
        # .. doc-marker-end: training-setup-separation

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    exit(main())
