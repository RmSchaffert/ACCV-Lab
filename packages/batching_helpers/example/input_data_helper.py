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


def insert_gt_data_noisy(
    data_gt, data_pred, indices_gt, indices_pred, noise_level, data_is_bboxes, negative_only
):
    for s in range(len(indices_gt)):
        for i in range(len(indices_gt[s])):
            i_pred = indices_pred[s][i]
            i_gt = indices_gt[s][i]
            to_insert = data_gt[s][i_gt]
            # Don't use += as this will overwrite the original data
            to_add = torch.randn_like(to_insert) * noise_level
            if negative_only:
                to_add = -torch.abs(to_add)
            to_insert = to_insert + to_add
            if data_is_bboxes:
                to_insert[..., 2:4] = torch.max(to_insert[..., 0:2], to_insert[..., 2:4])
            data_pred[s, i_pred, ...] = to_insert


def create_one_hot(classes_gt, num_classes):
    batch_size = len(classes_gt)
    res = []
    for s in range(batch_size):
        curr_res = torch.zeros((classes_gt[s].shape[0], num_classes), dtype=torch.float32, device="cuda:0")
        for i_target, target_cgt in enumerate(classes_gt[s]):
            curr_res[i_target, target_cgt] = 1.0
        res.append(curr_res)
    return res


def insert_existence_noisy(pred_existence, indices_pred, noise_level):
    batch_size = pred_existence.shape[0]
    for s in range(batch_size):
        for i_p in indices_pred[s]:
            pred_existence[s, i_p] = 1.0 - torch.rand(1, dtype=torch.float32, device="cuda:0") * noise_level
