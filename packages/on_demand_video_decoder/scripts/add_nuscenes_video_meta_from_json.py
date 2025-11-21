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
import json
import argparse


def main():
    args = parse_args()

    nuscenes_root_dir = args.nuscenes_root
    nuscenes_version = args.nuscenes_version
    video_sub_dir = args.video_sub_dir

    # Inputs/outputs
    sample_data_file_in_path = os.path.join(nuscenes_root_dir, nuscenes_version, 'sample_data.json')
    sample_data_file_out_path = os.path.join(nuscenes_root_dir, nuscenes_version, 'sample_data_video.json')
    mapping_json_path = os.path.join(nuscenes_root_dir, video_sub_dir, 'image_to_video_mapping.json')

    # Load mapping
    with open(mapping_json_path, 'r') as f:
        image_to_video_map = json.load(f)

    # Load sample_data
    with open(sample_data_file_in_path, 'r') as f:
        json_sample_data = json.load(f)

    # Apply mapping: only update camera entries (.jpg) present in the mapping
    updated = 0
    for entry in json_sample_data:
        rel_path = entry.get('filename')
        if not rel_path or not rel_path.lower().endswith('.jpg'):
            continue
        mapping = image_to_video_map.get(rel_path)
        if mapping is None:
            continue
        # mapping['video_path'] is relative to the video output subdirectory (e.g., "video_samples").
        # Store video_filename including the videos folder so it's relative to the NuScenes root.
        entry['video_filename'] = os.path.join(video_sub_dir, mapping['video_path'])
        entry['video_frame'] = mapping['frame_index']
        updated += 1

    # Write output
    with open(sample_data_file_out_path, 'w') as f:
        json.dump(json_sample_data, f, indent=2)

    print(f'Updated entries: {updated}')
    print(f'Saved: {sample_data_file_out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Add NuScenes video metadata from mapping JSON')
    parser.add_argument(
        '--nuscenes_root',
        type=str,
        required=False,
        default='/data/nuscenes',
        help='Path to NuScenes dataset root directory',
    )
    parser.add_argument(
        '--nuscenes_version',
        type=str,
        required=False,
        default='v1.0-mini',
        help='NuScenes version subdirectory, e.g., v1.0-mini',
    )
    parser.add_argument(
        '--video_sub_dir',
        type=str,
        required=False,
        default='video_samples',
        help='Path to subdirectory containing the generated videos. Default is "video_samples"',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
