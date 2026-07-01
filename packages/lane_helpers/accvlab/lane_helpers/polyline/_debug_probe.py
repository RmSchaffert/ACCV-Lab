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

"""Debug-only CUDA probe for lane_helpers polyline extension."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import torch

from accvlab.lane_helpers import _polyline_sampling


def run_probe() -> Dict[str, Any]:
    """Import the extension and run a minimal CUDA interpolation smoke test."""
    payload: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "module_file": getattr(_polyline_sampling, "__file__", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_smoke": None,
        "cuda_smoke": None,
        "cuda_error": None,
    }

    points_cpu = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    distances_cpu = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    try:
        payload["cpu_smoke"] = _polyline_sampling.polyline_interpolation(
            points_cpu, distances_cpu, relative=False
        ).tolist()
    except Exception as exc:  # noqa: BLE001 - debug probe
        payload["cpu_smoke_error"] = repr(exc)

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        payload["cuda_error"] = "CUDA not available at probe time"
        return payload

    points_cuda = points_cpu.cuda()
    distances_cuda = distances_cpu.cuda()
    try:
        payload["cuda_smoke"] = _polyline_sampling.polyline_interpolation(
            points_cuda, distances_cuda, relative=False
        ).cpu().tolist()
        payload["device_capability"] = list(torch.cuda.get_device_capability(0))
        payload["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001 - debug probe
        payload["cuda_error"] = repr(exc)
        if torch.cuda.is_available():
            payload["device_capability"] = list(torch.cuda.get_device_capability(0))
            payload["device_name"] = torch.cuda.get_device_name(0)

    return payload
