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

"""Pytest hooks for lane_helpers CUDA debug collection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ON_FAILURE_COLLECTED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _maybe_collect_on_failure() -> None:
    global _ON_FAILURE_COLLECTED
    if _ON_FAILURE_COLLECTED:
        return

    artifact_dir = os.environ.get("ACCVLAB_DEBUG_ARTIFACT_DIR")
    if not artifact_dir:
        return

    _ON_FAILURE_COLLECTED = True
    os.environ["ACCVLAB_LANE_HELPERS_CUDA_FAILED"] = "1"
    script = _repo_root() / "scripts" / "debug_lane_helpers_cuda.sh"
    if not script.is_file():
        return

    subprocess.run(
        [str(script), "collect", "on-failure"],
        check=False,
        cwd=str(_repo_root()),
    )


def pytest_configure(config) -> None:
    artifact_dir = os.environ.get("ACCVLAB_DEBUG_ARTIFACT_DIR")
    if artifact_dir:
        print(f"[lane_helpers debug] ACCVLAB_DEBUG_ARTIFACT_DIR={artifact_dir}", file=sys.stderr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    if "cuda" not in item.nodeid.lower():
        return
    _maybe_collect_on_failure()
