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

""" """

from ast import Str
import string

try:
    # Import native module
    from ._PyNvOnDemandDecoder import *  # noqa
    from enum import Enum

except ImportError:
    import distutils.sysconfig
    from os.path import join, dirname

    raise RuntimeError(
        "Failed to import native module _PyNvOnDemandDecoder! "
        f"Please check whether \"{join(dirname(__file__), '_PyNvOnDemandDecoder' + distutils.sysconfig.get_config_var('EXT_SUFFIX'))}\""  # noqa
        " exists and can find all library dependencies (CUDA, ffmpeg).\n"
        "On Unix systems, you can use `ldd` on the file to see whether it can find all dependencies.\n"
        "On Windows, you can use \"dumpbin /dependents\" in a Visual Studio command prompt or\n"
        "https://github.com/lucasg/Dependencies/releases."
    )


class Codec(Enum):
    h264 = 4
    hevc = 8
    av1 = 11
