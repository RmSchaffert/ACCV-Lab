/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>

#define THREADS_PER_BLOCK 1024

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

static constexpr dim3 THREADS_PER_BLOCK_BATCHED{8, 128, 1};

static __device__ __forceinline__ float atomicMax(float* addr, float val) {
    unsigned int old = __float_as_uint(*addr), assumed;
    do {
        assumed = old;
        if (__uint_as_float(old) >= val) break;
        old = atomicCAS((unsigned int*)addr, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(old);
}

static __device__ __forceinline__ void draw_gaussian(float* heatmap, int left, int right, int top, int bottom,
                                                     int map_stride, int x, int y, int width, float var,
                                                     float k_scale) {
    const float var_inv = 1.0f / var;
    for (int i = -top; i < bottom; ++i) {
        const float ii = static_cast<float>(i * i);
        for (int j = -left; j < right; ++j) {
            const float jj = static_cast<float>(j * j);
            const int map_pos = map_stride + (y + i) * width + x + j;
            const float gaussian_v = expf(-(ii + jj) * var_inv) * k_scale;
            atomicMax(heatmap + map_pos, gaussian_v);
        }
    }
}

static __global__ void draw_heatmap_cuda_kernel(float* heatmap, const int* centers, const int* radii,
                                                const int* heatmap_idxes, int height, int width,
                                                float diameter_to_sigma_factor, float k_scale,
                                                int num_targets, int num_heatmaps) {
    CUDA_1D_KERNEL_LOOP(index, num_targets) {
        int x = centers[index * 2];
        int y = centers[index * 2 + 1];
        int radius = radii[index];
        int heatmap_idx = heatmap_idxes[index];

        int diameter = 2 * radius + 1;
        float sigma = static_cast<float>(diameter) / diameter_to_sigma_factor;
        float var = 2.0f * sigma * sigma;

        int left = min(x, radius);
        int right = min(width - x, radius + 1);
        int top = min(y, radius);
        int bottom = min(height - y, radius + 1);

        int map_stride = heatmap_idx * height * width;

        draw_gaussian(heatmap, left, right, top, bottom, map_stride, x, y, width, var, k_scale);
    }
}

static __global__ void draw_heatmap_batched_cuda_kernel(float* heatmap, const int* centers, const int* radii,
                                                        const int* nums_targets, int height, int width,
                                                        int max_num_targets, float diameter_to_sigma_factor,
                                                        float k_scale, int batch_size,
                                                        const int max_num_classes, const int* labels) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    const int target = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample < batch_size && target < nums_targets[sample]) {
        const int index = sample * max_num_targets + target;
        const int x = centers[index * 2];
        const int y = centers[index * 2 + 1];
        const int radius = radii[index];

        const int diameter = 2 * radius + 1;
        const float sigma = static_cast<float>(diameter) / diameter_to_sigma_factor;
        const float var = 2.0f * sigma * sigma;

        const int left = min(x, radius);
        const int right = min(width - x, radius + 1);
        const int top = min(y, radius);
        const int bottom = min(height - y, radius + 1);

        int map_stride = sample * height * width;
        if (max_num_classes > 0 && labels != nullptr) {
            const int label = labels[index];
            assert(label >= 0 && label < max_num_classes);

            map_stride = sample * max_num_classes * height * width + label * height * width;
        }

        draw_gaussian(heatmap, left, right, top, bottom, map_stride, x, y, width, var, k_scale);
    }
}