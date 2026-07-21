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

import pytest
import torch

from accvlab.lane_helpers import frenet

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_straight_reference_lane_transform(device: str, dtype: torch.dtype):
    reference_lane_points = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    points_to_transform = torch.tensor(
        [[0.0, -1.0], [1.0, 2.0], [3.0, -1.0]],
        device=device,
        dtype=dtype,
    )

    result = frenet.transform(reference_lane_points, points_to_transform)

    expected = torch.tensor(
        [[0.0, -1.0], [1.0, 2.0], [2.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    assert torch.allclose(result, expected, atol=1e-5, rtol=0.0)

@pytest.mark.parametrize("device", DEVICES)
def test_multi_segment_reference_lane_transform_with_geometry_and_segment_normals(device: str):
    reference_lane_points = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]],
        device=device,
        dtype=torch.float32,
    )
    points_to_transform = torch.tensor(
        [
            [0.75, 0.25],
            [0.25, 0.75],
            [2.0, 1.0],
            [0.0, -1.0],
            [0.0, 3.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    result = frenet.transform(reference_lane_points, points_to_transform)

    inv_sqrt2 = 2.0 ** -0.5
    sqrt2 = 2.0**0.5
    # Each segment has unit tangent (±1, 1) / sqrt(2), so projected lengths and
    # normal distances on these 45-degree segments are simple multiples of 1 / sqrt(2).
    expected = torch.tensor(
        [
            # Inside the first segment, below the line: s is the projection length, d uses the left normal.
            [inv_sqrt2, -0.5 * inv_sqrt2],
            # Inside the first segment, above the line: same projection length, opposite signed distance.
            [inv_sqrt2, 0.5 * inv_sqrt2],
            # At the joint: s is one segment length, d uses the averaged normal pointing left.
            [sqrt2, -1.0],
            # Before the first segment: projection clamps to the start point.
            [0.0, -inv_sqrt2],
            # After the last segment: projection clamps to the final point.
            [2.0 * sqrt2, -inv_sqrt2],
        ],
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected, atol=1e-5, rtol=0.0)

    # Passing the opposite per-segment normals keeps the selected projections and s-values unchanged,
    # but flips the sign of every signed normal distance.
    opposite_segment_normals = torch.tensor(
        [[inv_sqrt2, -inv_sqrt2], [inv_sqrt2, inv_sqrt2]],
        device=device,
        dtype=torch.float32,
    )
    result_with_opposite_segment_normals = frenet.transform(
        reference_lane_points,
        points_to_transform,
        per_segment_normals=opposite_segment_normals,
    )
    expected_with_opposite_segment_normals = expected.clone()
    expected_with_opposite_segment_normals[:, 1] *= -1.0
    assert torch.allclose(
        result_with_opposite_segment_normals,
        expected_with_opposite_segment_normals,
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_frenet_transform_accepts_non_contiguous_inputs(device: str):
    reference_lane_storage = torch.tensor(
        [[0.0, 2.0], [0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    points_storage = torch.tensor(
        [[0.0, 1.0, 3.0], [-1.0, 2.0, -1.0]],
        device=device,
        dtype=torch.float32,
    )
    reference_lane_points = reference_lane_storage.transpose(0, 1)
    points_to_transform = points_storage.transpose(0, 1)
    assert not reference_lane_points.is_contiguous()
    assert not points_to_transform.is_contiguous()

    result = frenet.transform(reference_lane_points, points_to_transform)

    expected = torch.tensor(
        [[0.0, -1.0], [1.0, 2.0], [2.0, -1.0]],
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_frenet_transform_validates_dtype(device: str):
    reference_lane_points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    points_to_transform = torch.tensor(
        [[0.0, 0.0]],
        device=device,
        dtype=torch.float64,
    )

    with pytest.raises(RuntimeError, match="same dtype"):
        frenet.transform(reference_lane_points, points_to_transform)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_frenet_transform_rejects_mixed_cpu_cuda_inputs():
    reference_lane_points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    points_to_transform = torch.tensor([[0.0, 0.0]], device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="same device"):
        frenet.transform(reference_lane_points, points_to_transform)


def test_cpu_frenet_transform_rejects_low_precision_dtypes():
    for dtype in (torch.float16, torch.bfloat16):
        reference_lane_points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
        points_to_transform = torch.tensor([[0.0, 0.0]], dtype=dtype)

        with pytest.raises(RuntimeError, match="float32 or float64 on CPU"):
            frenet.transform(reference_lane_points, points_to_transform)


@pytest.mark.parametrize("device", DEVICES)
def test_frenet_transform_uses_per_segment_normals(device: str):
    reference_lane_points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    points_to_transform = torch.tensor(
        [[0.5, 1.0], [1.0, 1.0], [1.5, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    per_segment_normals = torch.tensor(
        [[0.0, -1.0], [0.0, -1.0]],
        device=device,
        dtype=torch.float32,
    )

    result = frenet.transform(
        reference_lane_points,
        points_to_transform,
        per_segment_normals=per_segment_normals,
    )

    expected = torch.tensor(
        [[0.5, -1.0], [1.0, -1.0], [1.5, -1.0]],
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_frenet_transform_interpolates_per_vertex_normals(device: str):
    reference_lane_points = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    points_to_transform = torch.tensor(
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    per_vertex_normals = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
        device=device,
        dtype=torch.float32,
    )

    result = frenet.transform(
        reference_lane_points,
        points_to_transform,
        per_vertex_normals=per_vertex_normals,
    )

    inv_sqrt2 = 2.0 ** -0.5
    expected = torch.tensor(
        [
            # At the first vertex, the start normal is used.
            [0.0, 1.0],
            # Halfway along the first segment, the interpolated normal is normalized from (0.5, 0.5).
            [1.0, inv_sqrt2],
            # At the shared vertex, the vertex normal is orthogonal to the point offset.
            [2.0, 0.0],
            # Halfway along the second segment, the interpolated normal is normalized from (0.5, -0.5).
            [3.0, -inv_sqrt2],
            # At the last vertex, the end normal points opposite to the point offset.
            [4.0, -1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected, atol=1e-5, rtol=0.0)


def test_frenet_transform_rejects_both_normal_modes():
    reference_lane_points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    points_to_transform = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    normals = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="either per_segment_normals or per_vertex_normals"):
        frenet.transform(
            reference_lane_points,
            points_to_transform,
            per_segment_normals=normals,
            per_vertex_normals=normals,
        )


if __name__ == "__main__":
    pytest.main([__file__])