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
from accvlab.draw_heatmap import draw_heatmap
from input_data import (
    device,
    centers_list,
    radii_list,
    HEATMAP_SIZE,
    diameter_to_sigma_factor,
    k_scale,
)


def draw_heatmap_example():
    """
    This example shows how to draw heatmaps for concatenated inputs.

    Args:
        centers_list: List[torch.Tensor]
        radii_list: List[torch.Tensor]
        HEATMAP_SIZE: [H, W]
        diameter_to_sigma_factor: float
        k_scale: float
    """
    num_of_heatmaps = len(centers_list)

    centers = torch.cat(centers_list, dim=0)
    radii = torch.cat(radii_list, dim=0)
    heatmaps = torch.zeros(
        (num_of_heatmaps, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=torch.float32, device=device
    )

    # because we concat all samples, we need to record the index of the heatmap for each bounding box
    heatmap_idxes = torch.tensor(
        [i for i, sublist in enumerate(centers_list) for _ in sublist],
        device=device,
        dtype=torch.int32,
    )
    draw_heatmap(heatmaps, centers, radii, heatmap_idxes, diameter_to_sigma_factor, k_scale)
    return heatmaps


if __name__ == "__main__":
    heatmaps = draw_heatmap_example()
    print(heatmaps.shape)
