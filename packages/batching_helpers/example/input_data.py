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
import input_data_helper

# GT data
rects_gt = [
    torch.tensor(
        [
            [10, 20, 100, 200],
            [50, 10, 100, 100],
            [30, 30, 65, 75],
        ],
        dtype=torch.float32,
        device="cuda:0",
    ),
    torch.tensor(
        [
            [20, 20, 300, 200],
            [150, 100, 160, 200],
        ],
        dtype=torch.float32,
        device="cuda:0",
    ),
    torch.tensor([[]], dtype=torch.float32, device="cuda:0"),
    torch.tensor(
        [
            [15, 25, 150, 250],
            [20, 30, 400, 50],
            [100, 200, 120, 220],
        ],
        dtype=torch.float32,
        device="cuda:0",
    ),
]

classes_gt = [
    torch.tensor([0, 2, 4], dtype=torch.int64, device="cuda:0"),
    torch.tensor([3, 1], dtype=torch.int64, device="cuda:0"),
    torch.tensor([], dtype=torch.int64, device="cuda:0"),
    torch.tensor([4, 2, 3], dtype=torch.int64, device="cuda:0"),
]
classes_gt_onehot = input_data_helper.create_one_hot(classes_gt, 10)

weights_gt = [
    torch.tensor([1.0, 0.5, 0.7], dtype=torch.float32, device="cuda:0"),
    torch.tensor([0.3, 0.5], dtype=torch.float32, device="cuda:0"),
    torch.tensor([], dtype=torch.float32, device="cuda:0"),
    torch.tensor([0.5, 1.0, 1.0], dtype=torch.float32, device="cuda:0"),
]

torch.manual_seed(10)

# Predictions (will be adjusted to match GT)
rects_pred = torch.rand(4, 100, 4, device="cuda:0", dtype=torch.float32) * 300
rects_pred[2:] += rects_pred[:2]

classes_pred_onehot = torch.rand(4, 100, 10, device="cuda:0", dtype=torch.float32)
pred_existence = torch.rand(4, 100, device="cuda:0", dtype=torch.float32)

# Matches between GT and predictions
indices_gt = [[1, 2, 0], [0, 1], [], [0, 2, 1]]
indices_pred = [[50, 21, 75], [20, 5], [], [42, 16, 8]]

# Insert noisy GT into predictions to simulate reasonable network output
input_data_helper.insert_gt_data_noisy(rects_gt, rects_pred, indices_gt, indices_pred, 5.0, True, False)
input_data_helper.insert_gt_data_noisy(
    classes_gt_onehot, classes_pred_onehot, indices_gt, indices_pred, 0.05, False, True
)
input_data_helper.insert_existence_noisy(pred_existence, indices_pred, 0.2)
