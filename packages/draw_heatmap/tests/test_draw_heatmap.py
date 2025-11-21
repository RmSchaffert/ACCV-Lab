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

import torch
import pytest
from accvlab.draw_heatmap import draw_heatmap

NUM_HEATMAPS = 48
HEATMAP_SIZE = [20, 50]
IMG_SHAPE = [320, 800, 3]
OUT_SIZE_FACTOR = 16
MAX_NUM_TARGET = 50
MAX_BOX_SIZE = 120

from _test_helpers import (
    generate_gt_bboxes,
    get_centers_and_radii,
    get_centers_and_radii_multiple_samples,
    get_heatmap_multiple_samples,
)


def test_draw_heatmap():
    # Gaussian 2d kernel parameters
    diameter_to_sigma_factor = 6
    k_scale = 0.8

    # Generate ground truth boxes
    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    gt_centers2d_list, gt_bboxes2d_list = generate_gt_bboxes(
        device, NUM_HEATMAPS, MAX_NUM_TARGET, IMG_SHAPE, MAX_BOX_SIZE
    )

    # Existing Implementation
    centers_list, radii_list = get_centers_and_radii_multiple_samples(
        gt_centers2d_list, gt_bboxes2d_list, OUT_SIZE_FACTOR
    )
    radii_list = [t.cpu().numpy().tolist() for t in radii_list]

    heatmaps_ref = get_heatmap_multiple_samples(
        centers_list,
        radii_list,
        k_scale=k_scale,
        diameter_to_sigma_factor=diameter_to_sigma_factor,
        heatmap_size=HEATMAP_SIZE,
    )

    # GPU Kernel Implementation

    # Concatenate the centers and radii for all samples
    gt_centers2d = torch.cat(gt_centers2d_list, dim=0)
    gt_bboxes2d = torch.cat(gt_bboxes2d_list, dim=0)

    # Use helper functions to get the centers and radii (not part of the GPU Kernel Implementation)
    centers, radii = get_centers_and_radii(gt_centers2d, gt_bboxes2d, OUT_SIZE_FACTOR)

    # Set up the heatmap indices
    heatmap_idxes = torch.tensor(
        [i for i, sublist in enumerate(gt_centers2d_list) for _ in sublist],
        device=device,
        dtype=torch.int32,
    )
    # Initialize the heatmaps
    heatmaps = gt_centers2d.new_zeros(NUM_HEATMAPS, HEATMAP_SIZE[0], HEATMAP_SIZE[1])

    # Perform the actual drawing of the heatmaps
    draw_heatmap(heatmaps, centers, radii, heatmap_idxes, diameter_to_sigma_factor, k_scale)

    # Compare the results
    heatmaps_ref = torch.stack(heatmaps_ref)
    diff = heatmaps - heatmaps_ref
    err = diff * diff
    assert err.mean() < 1e-3, "Draw heatmaps pytorch v.s. cuda kernel failed"


if __name__ == "__main__":
    pytest.main([__file__])
