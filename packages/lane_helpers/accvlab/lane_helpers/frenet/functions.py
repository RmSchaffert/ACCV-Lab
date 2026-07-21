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

import torch

from .. import _frenet


def transform(
    reference_lane_points: torch.Tensor,
    points_to_transform: torch.Tensor,
    *,
    per_segment_normals: torch.Tensor | None = None,
    per_vertex_normals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform Cartesian points to Frenet coordinates along a reference lane.

    Args:
        reference_lane_points: CPU or CUDA tensor with shape ``(num_reference_points, 2)``.
        points_to_transform: CPU or CUDA tensor with shape ``(num_points, 2)`` on the same device and
            with the same dtype as ``reference_lane_points``.
        per_segment_normals: Optional tensor with shape ``(num_reference_points - 1, 2)`` on the same
            device and with the same dtype as ``reference_lane_points``. Mutually exclusive with
            ``per_vertex_normals``.
        per_vertex_normals: Optional tensor with shape ``(num_reference_points, 2)`` on the same device
            and with the same dtype as ``reference_lane_points``. Normals are linearly interpolated
            along each segment. Mutually exclusive with ``per_segment_normals``.
            If neither normal tensor is provided, segment normals are computed from the reference-lane
            geometry.

    Returns:
        Tensor with shape ``(num_points, 2)`` on the same device as ``points_to_transform``.
        The last dimension stores longitudinal ``s`` and signed lateral distance.
    """
    # The extension handles validation and dispatches to the CPU or CUDA implementation.
    result = _frenet.frenet_trafo(
        reference_lane_points,
        points_to_transform,
        per_segment_normals,
        per_vertex_normals,
    )
    return result
