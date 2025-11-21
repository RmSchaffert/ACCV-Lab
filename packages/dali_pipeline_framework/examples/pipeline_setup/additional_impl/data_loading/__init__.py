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

'''
This module contains helper functionality for loading the data from the NuScenes Dataset.
Note that this is not part of the core functionality of the package and is only
included for convenience. For other datasets (or data which is not considered here such as lidar point clouds)
the user is expected to implement their own data loading functionality.
'''

# Import classes from sibling files
from .nuscenes_2d_detection_data_provider import Nuscenes2DDetectionDataProvider
from .nuscenes_streampetr_data_provider import NuscenesStreamPETRDataProvider
from .nuscenes_data_converter import NuScenesDataConverter
from .nuscenes_data import NuScenesData, NuScenesDataSequence, NuScenesDataSample
from .nuscenes_reader import NuScenesReader
from .bbox_projector import BboxProjector

# Export all classes for easy access
__all__ = [
    'Nuscenes2DDetectionDataProvider',
    'NuscenesStreamPETRDataProvider',
    'NuScenesDataConverter',
    'NuScenesData',
    'NuScenesDataSequence',
    'NuScenesDataSample',
    'NuScenesReader',
    'BboxProjector',
]
