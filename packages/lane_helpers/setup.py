# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from skbuild import setup
from setuptools import find_namespace_packages

_ACCVLAB_BUILD_CONFIG_IMPORT_ERROR = """
#########################################################################################
# Missing build dependency: accvlab-build-config.                                       #
#                                                                                       #
# ACCV-Lab package builds normally use --no-build-isolation, so the shared build helper #
# must already be installed in the active environment. Install it first with:           #
#                                                                                       #
#     pip install <ACCV-Lab root>/build_config                                          #
#                                                                                       #
# and retry.                                                                            #
#                                                                                       #
# Alternatively, use <ACCV-Lab root>/scripts/package_manager.sh to install packages in  #
# the documented order.                                                                 #
#########################################################################################
"""

try:
    from accvlab_build_config import build_cmake_args, detect_cuda_info
except ModuleNotFoundError as exc:
    if exc.name != "accvlab_build_config":
        raise
    raise RuntimeError(_ACCVLAB_BUILD_CONFIG_IMPORT_ERROR) from exc

_cmake_args = build_cmake_args()

_debug_artifact_dir = os.environ.get("ACCVLAB_DEBUG_ARTIFACT_DIR")
if _debug_artifact_dir:
    _debug_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "cmake_args": _cmake_args,
        "cuda_info": detect_cuda_info(),
    }
    _debug_path = Path(_debug_artifact_dir) / "lane_helpers-setup.json"
    _debug_path.parent.mkdir(parents=True, exist_ok=True)
    _debug_path.write_text(json.dumps(_debug_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _debug_lines = ["[lane_helpers debug] build_cmake_args:", *[f"  {arg}" for arg in _cmake_args]]
    for line in _debug_lines:
        print(line, file=sys.stderr)
        print(line)


setup(
    name="accvlab.lane_helpers",
    description="Lane helper utilities for ACCV-Lab.",
    packages=find_namespace_packages(include=["accvlab.lane_helpers*"]),
    include_package_data=True,
    zip_safe=False,
    cmake_source_dir="ext_impl",
    cmake_install_dir="accvlab/lane_helpers",
    cmake_args=_cmake_args,
)
