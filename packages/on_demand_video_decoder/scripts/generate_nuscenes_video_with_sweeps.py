#!/usr/bin/env python3

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
import subprocess as sp
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from collections import defaultdict
import json


# -----------  environment checks -----------
def assert_ffmpeg_has_required_encoders(required_encoders=('libx265',)):
    """
    Ensure that the locally available FFmpeg binary supports the required encoders.
    Raises RuntimeError with a helpful message if requirements are not met.
    """
    try:
        result = sp.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            'FFmpeg executable not found in PATH. Please install FFmpeg compiled with libx265 support. '
            '\nPlease note that a fully functional FFmpeg is not provided in the ACCV-Lab Docker image. '
            'Please use a custom environment to run this script.'
        ) from e
    except sp.CalledProcessError as e:
        raise RuntimeError(
            'Failed to execute FFmpeg to inspect available encoders. Ensure FFmpeg is installed and callable.'
            '\nPlease note that a fully functional FFmpeg is not provided in the ACCV-Lab Docker image. '
            'Please use a custom environment to run this script.'
        ) from e

    output = result.stdout or ''
    missing = [enc for enc in required_encoders if enc not in output]
    if missing:
        raise RuntimeError(
            'Your FFmpeg build is missing required video encoders: '
            f"{', '.join(missing)}. Reinstall FFmpeg with these enabled "
            '(e.g., built with --enable-libx265).'
            '\nNote that a fully functional FFmpeg is not provided in the ACCV-Lab Docker image. '
            'Please use a custom environment to run this script.'
        )


# -----------  tool: FFmpeg write frame -----------
def images_to_video_ffmpeg(img_paths, out_mp4, fps, gop_size, interpolation_num_additional_frames):
    if not img_paths:
        return
    # read first frame to get resolution
    from PIL import Image

    w, h = Image.open(img_paths[0]).size

    cmd = [
        'ffmpeg',
        '-f',
        'rawvideo',
        '-vcodec',
        'rawvideo',
        '-s',
        f'{w}x{h}',
        '-pix_fmt',
        'bgr24',
        '-r',
        str(fps),
        '-i',
        '-',
        '-c:v',
        'libx265',
        '-x265-params',
        "lowdelay=1",
        '-g',
        str(gop_size),
        '-bf',
        '0',
        '-pix_fmt',
        'yuv420p',
        '-color_range',
        'mpeg',
        out_mp4,
    ]
    print(cmd)

    next_image = None
    pipe = sp.Popen(cmd, stdin=sp.PIPE)
    num_imgs = len(img_paths)
    for i in tqdm(range(num_imgs), desc=os.path.basename(out_mp4), leave=True):
        path = img_paths[i]
        has_next = i < num_imgs - 1
        if next_image is None:
            img = Image.open(path).convert('RGB')
        else:
            img = next_image
            next_image = None
        bgr = bytes(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        pipe.stdin.write(bgr)
        if interpolation_num_additional_frames > 0 and has_next:
            next_image = Image.open(img_paths[i + 1]).convert('RGB')
            interpolation_factor = interpolation_num_additional_frames + 1
            for j in range(interpolation_num_additional_frames):
                intermediate_img = Image.blend(img, next_image, (j + 1) / interpolation_factor)
                bgr = bytes(cv2.cvtColor(np.array(intermediate_img), cv2.COLOR_RGB2BGR))
                pipe.stdin.write(bgr)

    pipe.stdin.close()
    pipe.wait()


def main():
    # -----------  parse arguments -----------
    args = parse_args()
    NUSCENES_ROOT = args.nuscenes_root

    # -----------  validate environment -----------
    # Encoder requirement is driven by how videos are written below (libx265).
    assert_ffmpeg_has_required_encoders(('libx265',))

    # Output directory inside nuscenes root
    OUTPUT_DIR = os.path.join(NUSCENES_ROOT, args.video_sub_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Camera folders to scan
    cam_list = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]
    # cam_list = ['CAM_FRONT']

    # Helper: collect images under a directory (if it exists)
    def collect_images(dir_path):
        results = []  # list of (abs_path, rel_to_root)
        if not os.path.isdir(dir_path):
            return results
        for fname in os.listdir(dir_path):
            abs_p = os.path.join(dir_path, fname)
            if os.path.isfile(abs_p):
                rel_p = os.path.relpath(abs_p, NUSCENES_ROOT)
                results.append((abs_p, rel_p))
        return results

    # mapping: relative image path (from nuscenes_root) -> { video_path (relative to OUTPUT_DIR), frame_index }
    image_to_video_map = {}

    for cam in cam_list:
        # Gather from samples and sweeps
        samples_dir = os.path.join(NUSCENES_ROOT, 'samples', cam)
        sweeps_dir = os.path.join(NUSCENES_ROOT, 'sweeps', cam)

        images = []
        images.extend(collect_images(samples_dir))
        images.extend(collect_images(sweeps_dir))

        if not images:
            continue

        # Group by base name before the last '__' (folders should not affect grouping)
        groups = defaultdict(list)  # base_key -> list of (sort_key, abs_path, rel_path)
        for abs_p, rel_p in images:
            base_name = os.path.basename(rel_p)
            stem = os.path.splitext(base_name)[0]
            if '__' in stem:
                base_key = stem.rsplit('__', 1)[0]
                ts_part = stem.rsplit('__', 1)[1]
            else:
                base_key = stem
                ts_part = stem

            # Make sequence folder name camera-agnostic by removing any trailing "__CAM_*" token
            if '__CAM_' in base_key:
                base_key = base_key.split('__CAM_', 1)[0]

            # Sort key: prefer numeric timestamp if available, else use timestamp string
            if ts_part.isdigit():
                sort_key = (0, int(ts_part))
            else:
                sort_key = (1, ts_part)

            groups[base_key].append((sort_key, abs_p, rel_p))

        # Output layout: per-sequence (and per-part) directories containing camera videos

        # For each group, sort by timestamp, split by large gaps, and create videos
        for base_key, items in groups.items():
            items.sort(key=lambda x: (x[0], os.path.basename(x[2])))
            # Split into segments when numeric timestamp gaps exceed threshold
            GAP_THRESHOLD = 1_000_0000  # 1e7
            segments = []
            current_segment = []
            prev_ts = None
            for sk, abs_p, rel_p in items:
                if sk[0] == 0:  # numeric timestamp
                    ts = sk[1]
                    if prev_ts is not None and ts - prev_ts > GAP_THRESHOLD and current_segment:
                        segments.append(current_segment)
                        current_segment = []
                    prev_ts = ts
                else:
                    prev_ts = None
                current_segment.append((sk, abs_p, rel_p))
            if current_segment:
                segments.append(current_segment)

            multi_segments = len(segments) > 1
            for seg_idx, seg_items in enumerate(segments):
                img_paths_sorted = [abs_p for _sk, abs_p, _rel in seg_items]
                # Create sequence directory (with part suffix if segmented)
                seq_dir = (
                    os.path.join(OUTPUT_DIR, f"{base_key}__part{seg_idx}")
                    if multi_segments
                    else os.path.join(OUTPUT_DIR, base_key)
                )
                os.makedirs(seq_dir, exist_ok=True)
                # Store camera video within the sequence directory using only camera name
                out_mp4 = os.path.join(seq_dir, f"{cam}.mp4")

                images_to_video_ffmpeg(
                    img_paths_sorted,
                    out_mp4,
                    fps=args.fps,
                    gop_size=args.gop_size,
                    interpolation_num_additional_frames=args.interpolation_num_frames,
                )

                rel_video_path = os.path.relpath(out_mp4, OUTPUT_DIR)
                for frame_index_orig, (_sk, _abs_p, rel_p) in enumerate(seg_items):
                    frame_index = frame_index_orig * (args.interpolation_num_frames + 1)
                    image_to_video_map[rel_p] = {
                        'video_path': rel_video_path,
                        'frame_index': frame_index,
                    }

    # Write mapping JSON under OUTPUT_DIR
    mapping_path = os.path.join(OUTPUT_DIR, 'image_to_video_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(image_to_video_map, f, indent=2)
    print(f'Saved mapping: {mapping_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert NuScenes dataset to videos')
    parser.add_argument(
        '--nuscenes_root',
        type=str,
        required=True,
        help='Path to NuScenes dataset root directory',
    )
    parser.add_argument(
        '--video_sub_dir',
        type=str,
        required=False,
        default='video_samples',
        help=(
            'Path to output directory for generated videos (will be created if it does not exist). '
            'Default is "video_samples"',
        ),
    )
    parser.add_argument(
        '--fps', type=int, required=False, default=12, help='Frames per second for the generated videos'
    )
    parser.add_argument(
        '--gop_size', type=int, required=False, default=30, help='GOP size for the generated videos'
    )
    parser.add_argument(
        '--interpolation_num_frames',
        type=int,
        required=False,
        default=1,
        help=(
            'Number of additional frames to add between existing frames. Frames are added by linear '
            'interpolation. A value of 0 means no additional frames are added.'
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
