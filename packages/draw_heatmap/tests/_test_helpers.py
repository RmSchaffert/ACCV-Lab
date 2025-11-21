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

from _gaussian_drawing import draw_heatmap_gaussian


def get_centers_and_radii(centers, bboxes, out_size_factor):
    top_left = centers - bboxes[..., :2]
    bottom_right = bboxes[..., 2:] - centers
    distances = torch.cat([top_left, bottom_right], dim=-1)
    radii = torch.ceil(torch.min(distances, dim=-1)[0] / out_size_factor)
    radii_int = radii.to(torch.int32)
    radii_int[radii_int < 1] = 1
    centers_int = (centers / out_size_factor).to(torch.int32)
    return centers_int, radii_int


def get_centers_and_radii_multiple_samples(centers_list, bboxes_list, out_size_factor):
    bboxes_list_res = []
    centers_list_res = []
    for centers, bboxes in zip(centers_list, bboxes_list):
        centers, radii = get_centers_and_radii(centers, bboxes, out_size_factor)
        centers_list_res.append(centers)
        bboxes_list_res.append(radii)
    return centers_list_res, bboxes_list_res


def get_heatmap_single(centers, radii, k_scale, diameter_to_sigma_factor, heatmap_size):
    heatmap = torch.zeros(heatmap_size, device=centers.device)
    for center, r in zip(centers, radii):
        heatmap = draw_heatmap_gaussian(heatmap, center, r, k_scale, diameter_to_sigma_factor)
    return heatmap


def get_heatmap_multiple_samples(centers_list, radii_list, k_scale, diameter_to_sigma_factor, heatmap_size):
    heatmaps = []
    for centers, radii in zip(centers_list, radii_list):
        heatmaps.append(get_heatmap_single(centers, radii, k_scale, diameter_to_sigma_factor, heatmap_size))
    return tuple(heatmaps)


def get_heatmap_single_with_labels(
    centers, radii, labels, k_scale, diameter_to_sigma_factor, heatmap_size, max_num_classes
):
    heatmaps = torch.zeros(max_num_classes, heatmap_size[0], heatmap_size[1], device=centers.device)
    for center, r, label in zip(centers, radii, labels):
        heatmaps[label] = draw_heatmap_gaussian(heatmaps[label], center, r, k_scale, diameter_to_sigma_factor)
    return heatmaps


def get_heatmaps_multiple_samples_with_labels(
    centers_list, radii_list, labels_list, k_scale, diameter_to_sigma_factor, heatmap_size, max_num_classes
):
    heatmaps = []
    for centers, radii, labels in zip(centers_list, radii_list, labels_list):
        heatmaps.append(
            get_heatmap_single_with_labels(
                centers, radii, labels, k_scale, diameter_to_sigma_factor, heatmap_size, max_num_classes
            )
        )
    return tuple(heatmaps)


def generate_gt_bboxes(device, num_heatmaps, max_num_targets, img_shape, max_bbox_size):
    gt_centers2d_list = []
    gt_bboxes2d_list = []
    for _ in range(num_heatmaps):
        num_target = torch.randint(0, max_num_targets + 1, (1,))
        gt_centers_x = torch.rand(num_target, device=device) * img_shape[1]
        gt_centers_y = torch.rand(num_target, device=device) * img_shape[0]
        cur_centers2d = torch.stack([gt_centers_x, gt_centers_y], dim=1)
        gt_centers2d_list.append(cur_centers2d)

        center_to_border_dists = torch.rand(num_target, 4, device=device) * max_bbox_size / 2
        gt_bboxes_left_top = cur_centers2d - center_to_border_dists[:, 0:2]
        gt_bboxes_right_bottom = cur_centers2d + center_to_border_dists[:, 2:4]

        cur_bboxes2d = torch.cat([gt_bboxes_left_top, gt_bboxes_right_bottom], dim=1)
        gt_bboxes2d_list.append(cur_bboxes2d)

    return gt_centers2d_list, gt_bboxes2d_list


def generate_gt_bboxes_with_labels(
    device, num_heatmaps, max_num_targets, img_shape, max_bbox_size, max_num_classes
):
    gt_centers2d_list, gt_bboxes2d_list = generate_gt_bboxes(
        device, num_heatmaps, max_num_targets, img_shape, max_bbox_size
    )
    gt_labels_list = []
    for i in range(num_heatmaps):
        num_target = gt_centers2d_list[i].shape[0]
        gt_labels = torch.randint(0, max_num_classes, (num_target,), device=device)
        gt_labels_list.append(gt_labels)
    return gt_centers2d_list, gt_bboxes2d_list, gt_labels_list
