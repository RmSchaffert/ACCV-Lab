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
This script only benchmarks the performance of drawing all classes in one heatmap.
"""

import torch
import numpy as np
from accvlab.draw_heatmap import draw_heatmap, draw_heatmap_batched
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


def generate_centers_and_radii(cfg, device=torch.device("cuda", 0)):
    box_scale = 2  # hard codeed for now
    max_box_heatmap_size = max(
        1, min(cfg['heatmap_size']) / 2 / box_scale
    )  # this is the down sampled size of box in the heatmap
    centers_list = []
    radii_list = []
    for _ in range(cfg['batch_size']):
        num_target = torch.randint(0, cfg['max_num_target'] + 1, (1,))
        centers_x = torch.rand(num_target, device=device) * cfg['heatmap_size'][1]
        centers_y = torch.rand(num_target, device=device) * cfg['heatmap_size'][0]
        centers = torch.stack([centers_x, centers_y], dim=1)
        radii = torch.rand(num_target, device=device) * max_box_heatmap_size
        radii = torch.clamp(radii, 1)
        centers_list.append(centers.to(torch.int32))
        radii_list.append(radii.to(torch.int32))
    return centers_list, radii_list


def get_heatmap_single(heatmap, centers, radii, k_scale, diameter_to_sigma_factor):
    for center, r in zip(centers, radii):
        heatmap = _draw_heatmap_gaussian(heatmap, center, r, k_scale, diameter_to_sigma_factor)


def get_heatmap_pytorch(heatmaps, cfg, centers_list, radii_list):
    for i, (centers, radii) in enumerate(zip(centers_list, radii_list)):
        get_heatmap_single(heatmaps[i], centers, radii, cfg['k_scale'], cfg['diameter_to_sigma_factor'])
    return heatmaps


def benchmark_pytorch(cfg, device=torch.device("cuda", 0)):
    centers_list, radii_list = generate_centers_and_radii(cfg, device=device)
    radii_list = [t.cpu().numpy().tolist() for t in radii_list]
    heatmaps = torch.zeros(cfg['batch_size'], cfg['heatmap_size'][0], cfg['heatmap_size'][1], device=device)
    start_time = time.time()
    for _ in range(cfg['num_iter']):
        get_heatmap_pytorch(heatmaps, cfg, centers_list, radii_list)
    end_time = time.time()
    return (end_time - start_time) / cfg['num_iter']


def benchmark_package_flattened(cfg, device=torch.device("cuda", 0)):
    centers_list, radii_list = generate_centers_and_radii(cfg, device=device)

    centers = torch.cat(centers_list, dim=0)
    radii = torch.cat(radii_list, dim=0)
    heatmaps = torch.zeros(cfg['batch_size'], cfg['heatmap_size'][0], cfg['heatmap_size'][1], device=device)
    heatmap_idxes = torch.tensor(
        [i for i, sublist in enumerate(centers_list) for _ in sublist],
        device=device,
        dtype=torch.int32,
    )
    start_time = time.time()
    for _ in range(cfg['num_iter']):
        draw_heatmap(heatmaps, centers, radii, heatmap_idxes, cfg['diameter_to_sigma_factor'], cfg['k_scale'])
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / cfg['num_iter']


def benchmark_package_batched(cfg, device=torch.device("cuda", 0)):
    centers_list, radii_list = generate_centers_and_radii(cfg, device=device)

    centers = torch.zeros((cfg['batch_size'], cfg['max_num_target'], 2), dtype=torch.int32, device=device)
    radii = torch.zeros((cfg['batch_size'], cfg['max_num_target']), dtype=torch.int32, device=device)
    gt_nums_targets = torch.zeros((cfg['batch_size'],), dtype=torch.int32, device=device)
    for i, sublist in enumerate(centers_list):
        centers[i, : len(sublist)] = centers_list[i]
        radii[i, : len(sublist)] = radii_list[i]
        gt_nums_targets[i] = len(sublist)

    centers_rb = RaggedBatch(centers, sample_sizes=gt_nums_targets)
    radii_rb = RaggedBatch(radii, sample_sizes=gt_nums_targets)
    heatmaps = torch.zeros(
        (cfg['batch_size'], cfg['heatmap_size'][0], cfg['heatmap_size'][1]),
        dtype=torch.float32,
        device=device,
    )
    start_time = time.time()
    for _ in range(cfg['num_iter']):
        draw_heatmap_batched(heatmaps, centers_rb, radii_rb, cfg['diameter_to_sigma_factor'], cfg['k_scale'])
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / cfg['num_iter']


if __name__ == "__main__":
    torch.manual_seed(42)  # important. Make sure the inputs of pytorch and GPU kernel are the same
    cfg = {
        'batch_size': 48,
        'max_num_target': 50,
        'heatmap_size': [20, 50],
        'k_scale': 1,
        'diameter_to_sigma_factor': 6,
        'num_iter': 10,
    }

    print(f" ==== Heatmap size: {cfg['batch_size']}x{cfg['heatmap_size'][0]}x{cfg['heatmap_size'][1]} ==== ")
    device = torch.device("cuda", 0)
    pytorch_time = benchmark_pytorch(cfg, device=device)
    package_flattened_time = benchmark_package_flattened(cfg, device=device)
    package_batched_time = benchmark_package_batched(cfg, device=device)
    print(f"PyTorch time: {pytorch_time * 1000: .4f} ms")
    print(f"Package flattened inputs time: {package_flattened_time * 1000: .4f} ms")
    print(f"Package batched inputs time: {package_batched_time * 1000: .4f} ms")
