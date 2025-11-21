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

"""
This file contains the input data for the draw_heatmap example.
"""

import torch

device = "cuda:0"

HEATMAP_SIZE = [100, 100]
diameter_to_sigma_factor = 6
k_scale = 1

MAX_NUM_TARGET = 10  # max number of bounding boxes across a batch
MAX_NUM_CLASSES = 20  # max number of classes in the dataset

# The number of bounding boxes in across a batch is not the same
# So put each sample into a list, each list element is a tensor of shape (num_boxes, 2) containing the center coordinates (x, y) of the bounding boxes
# make sure the centers are in the range of the heatmap size

# 4 samples in this batch
centers_list = [
    torch.tensor([[2, 3], [67, 50], [21, 10]], dtype=torch.int32, device=device),  # 3 boxes in this sample
    torch.tensor([[4, 65], [54, 1]], dtype=torch.int32, device=device),  # 2 boxes in this sample
    torch.tensor([[5, 76]], dtype=torch.int32, device=device),  # 1 box in this sample
    torch.tensor([[76, 13]], dtype=torch.int32, device=device),  # 1 box in this sample
]
# The radius of the bounding boxes, each list element is a tensor of shape (num_boxes,) containing the radius of the gaussian corresponding to each bounding box
# The length of the radii_list is the same as the centers_list
# 4 samples in this batch
radii_list = [
    torch.tensor([4, 1, 5], dtype=torch.int32, device=device),
    torch.tensor([1, 10], dtype=torch.int32, device=device),
    torch.tensor([5], dtype=torch.int32, device=device),
    torch.tensor([9], dtype=torch.int32, device=device),
]

# The labels of the bounding boxes, each list element is a tensor of shape (num_boxes,) containing the class index of the gaussian corresponding to each bounding box
# The length of the labels_list is the same as the centers_list
# 4 samples in this batch
labels_list = [
    torch.tensor([4, 10, 2], dtype=torch.int32, device=device),
    torch.tensor([3, 1], dtype=torch.int32, device=device),
    torch.tensor([0], dtype=torch.int32, device=device),
    torch.tensor([7], dtype=torch.int32, device=device),
]
