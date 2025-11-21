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
This module contains helper functions for setting up the DALI pipeline for specific use-cases.
The functionality can be re-used, but is mainly intended to be used as a starting point for the user
to set up their own pipeline.
'''

from .helper_funcs import kill_subprocesses
from .stream_petr_pipeline import setup_dali_pipeline_stream_petr_train
from .stream_petr_config import (
    get_default_stream_petr_pipeline_config,
    add_stream_petr_arguments_to_parser,
    add_stream_petr_config_to_arguments,
)
from .object_detection_2d_pipeline import setup_dali_pipeline_2d_object_detection
from .object_detection_2d_config import (
    get_default_object_detection_2d_pipeline_config,
    add_object_detection_2d_arguments_to_parser,
    add_object_detection_2d_config_to_arguments,
)
