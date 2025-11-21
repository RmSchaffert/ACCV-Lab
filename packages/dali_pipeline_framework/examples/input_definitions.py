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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset directory as defined by docker. For local setups, set the directory. It should contain "datas" and "imgs" sub-folders. Each opf these sub-folders contains
# folders with the dataset name, which in turn contain the data / folder with images respectively.

nuscenes_root_dir = '/data/nuscenes'

can_bus_root_dir = '/data/nuscenes'

nuscenes_version = 'v1.0-mini'

nuscenes_preproc_file_base_name = '{}_preproc_{}.pkl'
