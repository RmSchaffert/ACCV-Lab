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

import types

from nuscenes.nuscenes import NuScenes

# import nuscenes.scripts.export_2d_annotations_as_json as export_2d_annotations_as_json
from nuscenes.scripts import export_2d_annotations_as_json

import input_definitions

if __name__ == "__main__":
    args = types.SimpleNamespace()
    args.dataroot = input_definitions.nuscenes_root_dir
    args.version = input_definitions.nuscenes_version
    args.filename = 'image_annotations.json'
    args.visibilities = ['', '1', '2', '3', '4']
    args.image_limit = -1

    # nusc is a global variable which is set inside the module if the module is executes as __main__. Instead,
    # we set it manually here
    export_2d_annotations_as_json.nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    export_2d_annotations_as_json.main(args)
