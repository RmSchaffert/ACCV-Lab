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

void draw_heatmap_batched_cuda(cudaStream_t stream, float* heatmap, const int* centers, const int* radii,
                               const int* nums_targets, int height, int width, int max_num_targets,
                               float diameter_to_sigma_factor, float k_scale, int batch_size,
                               const int max_num_classes = 0, const int* labels = nullptr);

void generate_centers_and_radii_with_labels(int batch_size, int height, int width, int max_num_target,
                                            int max_num_classes, std::vector<std::vector<int>>& centers_list,
                                            std::vector<std::vector<int>>& radii_list,
                                            std::vector<std::vector<int>>& labels_list) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> num_target_dist(0, max_num_target);
    std::uniform_real_distribution<float> coord_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> radius_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, max_num_classes - 1);

    float box_scale = 2.0f;
    float max_box_heatmap_size = std::max(1.0f, std::min(height, width) / 2.0f / box_scale);

    centers_list.clear();
    radii_list.clear();
    labels_list.clear();
    for (int batch = 0; batch < batch_size; ++batch) {
        int num_target = num_target_dist(gen);

        std::vector<int> centers;
        std::vector<int> radii;
        std::vector<int> labels;
        for (int i = 0; i < num_target; ++i) {
            float center_x = coord_dist(gen) * width;
            float center_y = coord_dist(gen) * height;

            centers.push_back(static_cast<int>(center_x));
            centers.push_back(static_cast<int>(center_y));
            labels.push_back(label_dist(gen));

            float radius = radius_dist(gen) * max_box_heatmap_size;
            radius = std::max(1.0f, radius);
            radii.push_back(static_cast<int>(radius));
        }

        centers_list.push_back(centers);
        radii_list.push_back(radii);
        labels_list.push_back(labels);
    }
}

int main() {
    const int batch_size = 48;
    const int height = 20;
    const int width = 50;
    const int max_num_target = 50;
    const float diameter_to_sigma_factor = 6.0f;
    const float k_scale = 1.0f;
    const int max_num_classes = 20;
    std::cout << "Benchmarking draw heatmap function for batched inputs with class labels..." << std::endl;
    std::cout << "  Heatmap size: " << batch_size << "x" << max_num_classes << "x" << height << "x" << width
              << std::endl;
    std::vector<std::vector<int>> centers_list, radii_list, labels_list;
    generate_centers_and_radii_with_labels(batch_size, height, width, max_num_target, max_num_classes,
                                           centers_list, radii_list, labels_list);

    std::vector<int> centers(batch_size * max_num_target * 2), radii(batch_size * max_num_target);
    std::vector<int> labels(batch_size * max_num_target);
    std::vector<int> nums_targets(batch_size);
    int offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        int num_targets = radii_list[i].size();
        std::copy(centers_list[i].begin(), centers_list[i].begin() + num_targets * 2,
                  centers.begin() + i * max_num_target * 2);
        std::copy(radii_list[i].begin(), radii_list[i].begin() + num_targets,
                  radii.begin() + i * max_num_target);
        std::copy(labels_list[i].begin(), labels_list[i].begin() + num_targets,
                  labels.begin() + i * max_num_target);
        nums_targets[i] = num_targets;
    }

    // allocate GPU memory
    float* d_heatmap;
    int *d_centers, *d_radii, *d_nums_targets, *d_labels;
    cudaMalloc(&d_heatmap, batch_size * max_num_classes * height * width * sizeof(float));
    cudaMalloc(&d_centers, batch_size * max_num_target * 2 * sizeof(int));
    cudaMalloc(&d_radii, batch_size * max_num_target * sizeof(int));
    cudaMalloc(&d_nums_targets, batch_size * sizeof(int));
    cudaMalloc(&d_labels, batch_size * max_num_target * sizeof(int));
    // copy data to GPU
    cudaMemcpy(d_centers, centers.data(), batch_size * max_num_target * 2 * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_radii, radii.data(), batch_size * max_num_target * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nums_targets, nums_targets.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), batch_size * max_num_target * sizeof(int), cudaMemcpyHostToDevice);

    /////////////////////////    benchmark here      ///////////////////
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // warm up
    draw_heatmap_batched_cuda(stream, d_heatmap, d_centers, d_radii, d_nums_targets, height, width,
                              max_num_target, diameter_to_sigma_factor, k_scale, batch_size, max_num_classes,
                              d_labels);

    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        draw_heatmap_batched_cuda(stream, d_heatmap, d_centers, d_radii, d_nums_targets, height, width,
                                  max_num_target, diameter_to_sigma_factor, k_scale, batch_size,
                                  max_num_classes, d_labels);
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
    cudaFree(d_nums_targets);
    cudaFree(d_labels);

    return 0;
}