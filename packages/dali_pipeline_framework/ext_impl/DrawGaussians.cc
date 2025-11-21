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

#include "DrawGaussians.h"

#include <math.h>
#include <array>
#include <cstring>
#include <iostream>

namespace custom_operators {

template <typename T>
static T clip(T value, T min, T max) {
    T res = std::min(std::max(min, value), max);
    return res;
}

static void draw_gaussian(float* image, int32_t height, int32_t width, float center_x, float center_y,
                          float radius, float k, float radius_to_sigma_factor) {
    // Drawing area
    const int32_t min_x = static_cast<int32_t>(std::floor(static_cast<float>(center_x) - radius));
    const int32_t max_x = static_cast<int32_t>(std::ceil(static_cast<float>(center_x) + radius));
    const int32_t min_y = static_cast<int32_t>(std::floor(static_cast<float>(center_y) - radius));
    const int32_t max_y = static_cast<int32_t>(std::ceil(static_cast<float>(center_y) + radius));

    // If the drawing area is completely outside the image, no need to draw anything
    if ((max_x < 0 || min_x >= width) || (max_y < 0 || min_y >= height)) {
        return;
    }

    // Clipping of drawing area to image
    const int32_t min_x_clipped = clip(min_x, 0, width - 1);
    const int32_t max_x_clipped = clip(max_x, 0, width - 1);
    const int32_t min_y_clipped = clip(min_y, 0, height - 1);
    const int32_t max_y_clipped = clip(max_y, 0, height - 1);

    // Sigma computation
    const float sigma = radius * radius_to_sigma_factor;
    const float sigma_sqr_times_2_inv = 1.0f / (2.0f * sigma * sigma);

    for (int32_t y = min_y_clipped; y <= max_y_clipped; ++y) {
        const size_t index_line = static_cast<size_t>(y) * static_cast<size_t>(width);

        // Add a half pixel as the coordinates are (implicitly) defined as (0, 0) being the upper left corner of the upper left pixel, and we sample the center of the pixel
        int32_t diff_y_sqr = y - center_y;
        diff_y_sqr *= diff_y_sqr;

        for (int32_t x = min_x_clipped; x <= max_x_clipped; ++x) {
            const size_t index = index_line + static_cast<size_t>(x);

            // Add a half pixel as the coordinates are (implicitly) defined as (0, 0) being the upper left corner of the upper left pixel, and we sample the center of the pixel
            int32_t diff_x_sqr = x - center_x;
            diff_x_sqr *= diff_x_sqr;

            const float val =
                k * std::exp(-static_cast<float>(diff_y_sqr + diff_x_sqr) * sigma_sqr_times_2_inv);

            float& pixel = image[index];
            pixel = std::max(pixel, val);
        }
    }
}

template <>
DrawGaussians<::dali::CPUBackend>::DrawGaussians(const ::dali::OpSpec& spec)
    : ::dali::Operator<::dali::CPUBackend>(spec) {
    _k_for_classes = spec.GetRepeatedArgument<float>("k_for_classes");
    _radius_to_sigma_factor = spec.GetArgument<float>("radius_to_sigma_factor");
}

template <>
DrawGaussians<::dali::CPUBackend>::~DrawGaussians() {}

template <>
void DrawGaussians<::dali::CPUBackend>::RunImpl(::dali::Workspace& ws) {
    // heatmap, active, slice_ids, centers, radii
    const auto& heatmap_in = ws.Input<::dali::CPUBackend>(0);
    const auto& active = ws.Input<::dali::CPUBackend>(1);
    const auto& slice_ids = ws.Input<::dali::CPUBackend>(2);
    const auto& centers = ws.Input<::dali::CPUBackend>(3);
    const auto& radii = ws.Input<::dali::CPUBackend>(4);

    auto& output = ws.Output<::dali::CPUBackend>(0);

    const int batch_size = radii.shape().num_samples();

    auto& thread_pool = ws.GetThreadPool();

    const int num_threads = thread_pool.NumThreads();

    for (int s = 0; s < batch_size; ++s) {
        thread_pool.AddWork(
            // Note that "s" has to be passed by value as it keeps changing as the loop adds more
            // workers. When the work is actually executed, the following holds: s == batch_size.
            // The other parameters are are either meant to be modified inside the workers (output),
            // or remain constant during thread pool creation & execution (remaining parameters),
            // and are therefore passed as references.
            [s, &heatmap_in, &active, &slice_ids, &centers, &radii, &output, this](int thread_id) {
                const size_t num_objects = radii.shape()[s][0];

                const auto& heatmap_shape = heatmap_in.shape()[s];
                const bool has_channels = heatmap_shape.size() == 3;

                size_t curr_dim = 0;
                const size_t num_channels = has_channels ? heatmap_shape[curr_dim++] : 1;
                const size_t height = heatmap_shape[curr_dim++];
                const size_t width = heatmap_shape[curr_dim];
                const size_t channel_size = height * width;
                const size_t total_size = num_channels * channel_size;

                const float* heatmap_in_sample = static_cast<const float*>(heatmap_in.raw_tensor(s));
                const bool* active_sample = static_cast<const bool*>(active.raw_tensor(s));
                const int32_t* slice_ids_sample = static_cast<const int32_t*>(slice_ids.raw_tensor(s));
                const int32_t* centers_sample = static_cast<const int32_t*>(centers.raw_tensor(s));
                const float* radii_sample = static_cast<const float*>(radii.raw_tensor(s));

                float* heatmap_out_sample = static_cast<float*>(output.raw_mutable_tensor(s));

                std::memcpy(heatmap_out_sample, heatmap_in_sample, total_size * sizeof(float));

                for (size_t i = 0; i < num_objects; ++i) {
                    if (!active_sample[i]) {
                        continue;
                    }
                    const float radius = radii_sample[i];

                    const int32_t class_id = slice_ids_sample[i];

                    float* image = heatmap_out_sample + channel_size * class_id;

                    if (class_id >= this->_k_for_classes.size()) {
                        DALI_FAIL(std::string("class_id for active sample (") + std::to_string(class_id) +
                                  std::string(") exceeds elements in k_for_classes (") +
                                  std::to_string(this->_k_for_classes.size()) + std::string(")."));
                    }

                    draw_gaussian(image, static_cast<int32_t>(height), static_cast<int32_t>(width),
                                  centers_sample[i * 2], centers_sample[i * 2 + 1], radius,
                                  this->_k_for_classes[class_id], this->_radius_to_sigma_factor);
                }
            });
    }
    thread_pool.RunAll();
}

}  // namespace custom_operators

DALI_REGISTER_OPERATOR(draw_gaussians, ::custom_operators::DrawGaussians<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(draw_gaussians)
    .DocStr("Draw heat map using gaussians for each object")
    .NumInput(5)
    .NumOutput(1)
    .AddArg(
        "k_for_classes",
        "Weigths for the gaussians for each class ID, where the calss IDs correspond to indices in the array",
        ::dali::DALIDataType::DALI_FLOAT_VEC)
    .AddArg("radius_to_sigma_factor", "Factor used when computaing the sigma given an object radius",
            ::dali::DALIDataType::DALI_FLOAT);
