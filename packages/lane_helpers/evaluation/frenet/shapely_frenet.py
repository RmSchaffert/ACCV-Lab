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

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely import line_locate_point
from shapely.geometry import LineString


@dataclass(frozen=True)
class ShapelyFrenetReference:
    """Geometry reused by each benchmark call."""

    line_string: LineString
    reference_lane_points: np.ndarray
    cumulative_segment_ends: np.ndarray
    segment_normals: np.ndarray


def prepare_reference_lane(reference_lane_points: np.ndarray) -> ShapelyFrenetReference:
    """Precompute arc lengths and geometry-derived segment normals."""
    segments = np.diff(reference_lane_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    segment_tangents = segments / segment_lengths[:, None]
    segment_normals = np.stack((-segment_tangents[:, 1], segment_tangents[:, 0]), axis=1)

    return ShapelyFrenetReference(
        line_string=LineString(reference_lane_points),
        reference_lane_points=reference_lane_points,
        cumulative_segment_ends=np.cumsum(segment_lengths),
        segment_normals=segment_normals,
    )


def transform(
    reference: ShapelyFrenetReference,
    points_to_transform: np.ndarray,
    *,
    shapely_points: Any,
) -> np.ndarray:
    """Transform clean benchmark inputs to longitudinal and lateral coordinates."""
    longitudinal = np.asarray(line_locate_point(reference.line_string, shapely_points))
    segment_indices = np.searchsorted(
        reference.cumulative_segment_ends,
        longitudinal,
        side="right",
    )
    segment_indices = np.clip(segment_indices, 0, len(reference.segment_normals) - 1)

    offsets = points_to_transform - reference.reference_lane_points[segment_indices]
    selected_normals = reference.segment_normals[segment_indices]
    lateral = np.sum(offsets * selected_normals, axis=1)
    return np.column_stack((longitudinal, lateral))


def transform_unprepared(
    reference_lane_points: np.ndarray,
    points_to_transform: np.ndarray,
    *,
    shapely_points: Any,
) -> np.ndarray:
    """Transform points while rebuilding the reference geometry."""
    reference = prepare_reference_lane(reference_lane_points)
    return transform(
        reference,
        points_to_transform,
        shapely_points=shapely_points,
    )
