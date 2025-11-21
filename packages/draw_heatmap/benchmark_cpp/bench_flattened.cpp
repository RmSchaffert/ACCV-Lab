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

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

void draw_heatmap_cuda(cudaStream_t stream, float* heatmap, const int* centers, const int* radii,
                       const int* heatmap_idxes, int height, int width, float diameter_to_sigma_factor,
                       float k_scale, int num_targets, int num_heatmaps);

void generate_centers_and_radii(int batch_size, int height, int width, int max_num_target,
                                std::vector<std::vector<int>>& centers_list,
                                std::vector<std::vector<int>>& radii_list) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> num_target_dist(0, max_num_target);
    std::uniform_real_distribution<float> coord_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> radius_dist(0.0f, 1.0f);

    float box_scale = 2.0f;
    float max_box_heatmap_size = std::max(1.0f, std::min(height, width) / 2.0f / box_scale);

    centers_list.clear();
    radii_list.clear();

    for (int batch = 0; batch < batch_size; ++batch) {
        int num_target = num_target_dist(gen);

        std::vector<int> centers;
        std::vector<int> radii;

        for (int i = 0; i < num_target; ++i) {
            float center_x = coord_dist(gen) * width;
            float center_y = coord_dist(gen) * height;

            centers.push_back(static_cast<int>(center_x));
            centers.push_back(static_cast<int>(center_y));

            float radius = radius_dist(gen) * max_box_heatmap_size;
            radius = std::max(1.0f, radius);
            radii.push_back(static_cast<int>(radius));
        }

        centers_list.push_back(centers);
        radii_list.push_back(radii);
    }
}

int main() {
    const int batch_size = 48;
    const int height = 20;
    const int width = 50;
    const int max_num_target = 50;
    const float diameter_to_sigma_factor = 6.0f;
    const float k_scale = 1.0f;
    std::cout << "Benchmarking draw heatmap function for flattened inputs..." << std::endl;
    std::cout << "  Heatmap size: " << batch_size << "x" << height << "x" << width << std::endl;

    std::vector<std::vector<int>> centers_list, radii_list;
    generate_centers_and_radii(batch_size, height, width, max_num_target, centers_list, radii_list);
    int total_num_targets = 0;
    for (int i = 0; i < batch_size; ++i) {
        total_num_targets += radii_list[i].size();
    }
    std::vector<int> heatmap_idxes(total_num_targets, 0);

    // allocate GPU memory
    float* d_heatmap;
    int *d_centers, *d_radii, *d_heatmap_idxes;
    cudaMalloc(&d_heatmap, batch_size * height * width * sizeof(float));
    cudaMalloc(&d_centers, total_num_targets * 2 * sizeof(int));
    cudaMalloc(&d_radii, total_num_targets * sizeof(int));
    cudaMalloc(&d_heatmap_idxes, total_num_targets * sizeof(int));

    // copy data to GPU
    int offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        int num_targets = radii_list[i].size();
        cudaMemcpy(d_centers + offset * 2, centers_list[i].data(), num_targets * 2 * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_radii + offset, radii_list[i].data(), num_targets * sizeof(int), cudaMemcpyHostToDevice);
        std::fill(heatmap_idxes.begin() + offset, heatmap_idxes.begin() + offset + num_targets, i);
        offset += num_targets;
    }
    cudaMemcpy(d_heatmap_idxes, heatmap_idxes.data(), total_num_targets * sizeof(int),
               cudaMemcpyHostToDevice);

    /////////////////////////    benchmark here      ///////////////////
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // warm up
    draw_heatmap_cuda(stream, d_heatmap, d_centers, d_radii, d_heatmap_idxes, height, width,
                      diameter_to_sigma_factor, k_scale, total_num_targets, batch_size);

    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        draw_heatmap_cuda(stream, d_heatmap, d_centers, d_radii, d_heatmap_idxes, height, width,
                          diameter_to_sigma_factor, k_scale, total_num_targets, batch_size);
    }
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // convert to ms
    float duration_ms = duration.count() / 1000.0f;
    std::cout << "Benchmark completed:" << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Average time per iteration: " << duration_ms / num_iterations << " ms" << std::endl;

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_heatmap);
    cudaFree(d_centers);
    cudaFree(d_radii);
    cudaFree(d_heatmap_idxes);

    return 0;
}