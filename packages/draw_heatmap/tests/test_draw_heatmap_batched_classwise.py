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

from accvlab.draw_heatmap import draw_heatmap_batched
from accvlab.batching_helpers import RaggedBatch

from _test_helpers import (
    generate_gt_bboxes_with_labels,
    get_centers_and_radii,
    get_centers_and_radii_multiple_samples,
    get_heatmaps_multiple_samples_with_labels,
)

HEATMAP_SIZE = [20, 50]
IMG_SHAPE = [320, 800, 3]
OUT_SIZE_FACTOR = 16
MAX_NUM_TARGET = 50
MAX_BOX_SIZE = 120
BATCH_SIZE = 48
MAX_NUM_CLASSES = 20
VOC_DATA_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def test_draw_heatmap_batched_classwise():
    # Gaussian 2d kernel parameters
    diameter_to_sigma_factor = 6
    k_scale = 0.8

    # Generate ground truth boxes
    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    gt_centers2d_list, gt_bboxes2d_list, gt_labels_list = generate_gt_bboxes_with_labels(
        device, BATCH_SIZE, MAX_NUM_TARGET, IMG_SHAPE, MAX_BOX_SIZE, MAX_NUM_CLASSES
    )

    # Prepare GT boxes in the needed format

    # Initialize in a way that invalid bounding boxes would also be drawn to see if
    # only valid elements are drawn
    gt_centers2d = torch.ones((BATCH_SIZE, MAX_NUM_TARGET, 2), dtype=torch.float32, device=device) * 2
    gt_bboxes2d = torch.zeros((BATCH_SIZE, MAX_NUM_TARGET, 4), dtype=torch.float32, device=device)
    gt_labels = torch.zeros((BATCH_SIZE, MAX_NUM_TARGET), dtype=torch.int32, device=device)
    gt_nums_targets = torch.zeros(BATCH_SIZE, dtype=torch.int64, device=device)
    gt_bboxes2d[:, :, 0] = 1
    gt_bboxes2d[:, :, 1] = 1
    gt_bboxes2d[:, :, 2] = 3
    gt_bboxes2d[:, :, 3] = 3
    # Copy over the valid boxes
    for i in range(BATCH_SIZE):
        num_generated_targets = gt_centers2d_list[i].shape[0]
        gt_centers2d[i, 0:num_generated_targets, :] = gt_centers2d_list[i]
        gt_bboxes2d[i, 0:num_generated_targets, :] = gt_bboxes2d_list[i]
        gt_labels[i, 0:num_generated_targets] = gt_labels_list[i]
        gt_nums_targets[i] = num_generated_targets

    # Existing Implementation (using the lists as inputs)
    centers_list, radii_list = get_centers_and_radii_multiple_samples(
        gt_centers2d_list, gt_bboxes2d_list, OUT_SIZE_FACTOR
    )

    radii_list = [t.cpu().numpy().tolist() for t in radii_list]
    labels_list = [l.cpu().numpy().tolist() for l in gt_labels_list]

    heatmaps_ref = get_heatmaps_multiple_samples_with_labels(
        centers_list,
        radii_list,
        labels_list,
        k_scale=k_scale,
        diameter_to_sigma_factor=diameter_to_sigma_factor,
        heatmap_size=HEATMAP_SIZE,
        max_num_classes=MAX_NUM_CLASSES,
    )

    # GPU Kernel Implementation

    # Use helper functions to get the centers and radii (not part of the GPU Kernel Implementation)
    centers, radii = get_centers_and_radii(gt_centers2d, gt_bboxes2d, OUT_SIZE_FACTOR)
    # Initialize the heatmaps
    heatmaps = torch.zeros(
        (BATCH_SIZE, MAX_NUM_CLASSES, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=torch.float32, device=device
    )

    # Convert the centers and radii to ragged batch (the exoected input format)
    centers_rb = RaggedBatch(centers, sample_sizes=gt_nums_targets)
    radii_rb = RaggedBatch(radii, sample_sizes=gt_nums_targets)
    labels_rb = RaggedBatch(gt_labels, sample_sizes=gt_nums_targets)

    # Perform the actual drawing of the heatmaps
    draw_heatmap_batched(heatmaps, centers_rb, radii_rb, diameter_to_sigma_factor, k_scale, labels_rb)

    # Compare the results
    heatmaps_ref = torch.stack(heatmaps_ref)
    diff = heatmaps - heatmaps_ref
    err = diff * diff
    assert err.mean() < 1e-3, "Draw heatmaps classwise pytorch v.s. cuda batched kernel failed"


if __name__ == "__main__":
    pytest.main([__file__])
