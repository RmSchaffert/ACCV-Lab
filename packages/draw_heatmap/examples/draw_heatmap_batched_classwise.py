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
from accvlab.draw_heatmap import draw_heatmap_batched
from input_data import (
    device,
    centers_list,
    radii_list,
    labels_list,
    HEATMAP_SIZE,
    diameter_to_sigma_factor,
    k_scale,
    MAX_NUM_TARGET,
    MAX_NUM_CLASSES,
)
from accvlab.batching_helpers import RaggedBatch


def draw_heatmap_classwise_example_batch():
    """
    This example shows how to draw class-wise heatmaps for batched inputs.

    Args:
        centers_list: List[torch.Tensor]
        radii_list: List[torch.Tensor]
        labels_list: List[torch.Tensor]
        HEATMAP_SIZE: [H, W]
        diameter_to_sigma_factor: float
        k_scale: float
    """
    batch_size = len(centers_list)

    # In order to support the batched input, we need to convert the input to the format of ragged batch
    centers = torch.zeros((batch_size, MAX_NUM_TARGET, 2), dtype=torch.int32, device=device)
    radii = torch.zeros((batch_size, MAX_NUM_TARGET), dtype=torch.int32, device=device)
    labels = torch.zeros((batch_size, MAX_NUM_TARGET), dtype=torch.int32, device=device)
    gt_nums_targets = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    # Copy over the valid boxes
    for i, sublist in enumerate(centers_list):
        centers[i, : len(sublist)] = centers_list[i]
        radii[i, : len(sublist)] = radii_list[i]
        labels[i, : len(sublist)] = labels_list[i]
        gt_nums_targets[i] = len(sublist)

    centers_rb = RaggedBatch(centers, sample_sizes=gt_nums_targets)
    radii_rb = RaggedBatch(radii, sample_sizes=gt_nums_targets)
    labels_rb = RaggedBatch(labels, sample_sizes=gt_nums_targets)
    heatmaps = torch.zeros(
        (batch_size, MAX_NUM_CLASSES, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=torch.float32, device=device
    )
    draw_heatmap_batched(heatmaps, centers_rb, radii_rb, diameter_to_sigma_factor, k_scale, labels=labels_rb)
    return heatmaps


if __name__ == "__main__":
    heatmaps = draw_heatmap_classwise_example_batch()
    print(heatmaps.shape)
