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
Benchmark the performance of the draw_heatmap package.
This script only benchmarks the performance of drawing each class in a separate heatmap.
"""

import torch
import numpy as np
from accvlab.draw_heatmap import draw_heatmap_batched
from accvlab.batching_helpers import RaggedBatch
import time


def _gaussian_2d(shape, sigma):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _draw_heatmap_gaussian(heatmap, center, radius, k, factor):
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / factor)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top : radius + bottom, radius - left : radius + right]
    ).to(heatmap.device, torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def generate_centers_and_radii_with_labels(cfg, device=torch.device("cuda", 0)):
    box_scale = 2
    max_box_heatmap_size = max(
        1, min(cfg['heatmap_size']) / 2 / box_scale
    )  # this is the down sampled size of box in the heatmap
    centers_list = []
    radii_list = []
    labels_list = []
    for _ in range(cfg['batch_size']):
        num_target = torch.randint(0, cfg['max_num_target'] + 1, (1,))
        centers_x = torch.rand(num_target, device=device) * cfg['heatmap_size'][1]
        centers_y = torch.rand(num_target, device=device) * cfg['heatmap_size'][0]
        centers = torch.stack([centers_x, centers_y], dim=1)
        radii = torch.rand(num_target, device=device) * max_box_heatmap_size
        radii = torch.clamp(radii, 1)
        labels = torch.randint(0, cfg['max_num_classes'], (num_target,), device=device)

        centers_list.append(centers.to(torch.int32))
        radii_list.append(radii.to(torch.int32))
        labels_list.append(labels.to(torch.int32))
    return centers_list, radii_list, labels_list


def get_heatmap_single(heatmap, centers, radii, labels, k_scale, diameter_to_sigma_factor):
    for center, r, label in zip(centers, radii, labels):
        heatmap[label] = _draw_heatmap_gaussian(heatmap[label], center, r, k_scale, diameter_to_sigma_factor)


def get_heatmap_pytorch(heatmaps, cfg, centers_list, radii_list, labels_list):
    for i, (centers, radii, labels) in enumerate(zip(centers_list, radii_list, labels_list)):
        get_heatmap_single(
            heatmaps[i], centers, radii, labels, cfg['k_scale'], cfg['diameter_to_sigma_factor']
        )
    return heatmaps


def benchmark_pytorch(cfg, device=torch.device("cuda", 0)):
    centers_list, radii_list, labels_list = generate_centers_and_radii_with_labels(cfg, device=device)
    radii_list = [t.cpu().numpy().tolist() for t in radii_list]
    heatmaps = torch.zeros(
        cfg['batch_size'],
        cfg['max_num_classes'],
        cfg['heatmap_size'][0],
        cfg['heatmap_size'][1],
        device=device,
    )
    start_time = time.time()
    for _ in range(cfg['num_iter']):
        get_heatmap_pytorch(heatmaps, cfg, centers_list, radii_list, labels_list)
    end_time = time.time()
    return (end_time - start_time) / cfg['num_iter']


def benchmark_package_batched_classwise(cfg, device=torch.device("cuda", 0)):
    centers_list, radii_list, labels_list = generate_centers_and_radii_with_labels(cfg, device=device)

    centers = torch.zeros((cfg['batch_size'], cfg['max_num_target'], 2), dtype=torch.int32, device=device)
    radii = torch.zeros((cfg['batch_size'], cfg['max_num_target']), dtype=torch.int32, device=device)
    labels = torch.zeros((cfg['batch_size'], cfg['max_num_target']), dtype=torch.int32, device=device)
    gt_nums_targets = torch.zeros((cfg['batch_size'],), dtype=torch.int32, device=device)
    for i, sublist in enumerate(centers_list):
        centers[i, : len(sublist)] = centers_list[i]
        radii[i, : len(sublist)] = radii_list[i]
        labels[i, : len(sublist)] = labels_list[i]
        gt_nums_targets[i] = len(sublist)

    centers_rb = RaggedBatch(centers, sample_sizes=gt_nums_targets)
    radii_rb = RaggedBatch(radii, sample_sizes=gt_nums_targets)
    labels_rb = RaggedBatch(labels, sample_sizes=gt_nums_targets)
    heatmaps = torch.zeros(
        (cfg['batch_size'], cfg['max_num_classes'], cfg['heatmap_size'][0], cfg['heatmap_size'][1]),
        dtype=torch.float32,
        device=device,
    )
    start_time = time.time()
    for _ in range(cfg['num_iter']):
        draw_heatmap_batched(
            heatmaps, centers_rb, radii_rb, cfg['diameter_to_sigma_factor'], cfg['k_scale'], labels_rb
        )
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / cfg['num_iter']


if __name__ == "__main__":
    torch.manual_seed(42)  # important
    cfg = {
        'batch_size': 48,
        'max_num_target': 50,
        'heatmap_size': [20, 50],
        'k_scale': 1,
        'diameter_to_sigma_factor': 6,
        'max_num_classes': 20,
        'num_iter': 10,
    }

    print(
        f" ==== Heatmap size: {cfg['batch_size']}x{cfg['max_num_classes']}x{cfg['heatmap_size'][0]}x{cfg['heatmap_size'][1]} ==== "
    )
    device = torch.device("cuda", 0)
    pytorch_time = benchmark_pytorch(cfg, device=device)
    package_batched_time = benchmark_package_batched_classwise(cfg, device=device)
    print(f"PyTorch classwise time: {pytorch_time * 1000: .4f} ms")
    print(f"Package batched classwise time: {package_batched_time * 1000: .4f} ms")
