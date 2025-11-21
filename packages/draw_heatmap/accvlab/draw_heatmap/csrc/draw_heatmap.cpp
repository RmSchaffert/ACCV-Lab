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

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr const char* DOC_DRAW_HEATMAP = R"doc(
A function that draws Gaussian heatmaps based on centers and radii.

:gpu:

Args:
    heatmaps:
        The heatmaps to be drawn. This is a tensor of type `float32` with shape `(num_heatmaps, height, width)`.
    centers:
        The centers of each Gaussian kernel. This is a tensor of type `int32` with shape `(num_targets, 2)`.
    radii:
        The radii of each Gaussian kernel. This is a tensor of type `int32` with shape `(num_targets,)`.
    heatmap_idxes:
        The heatmap indices of each Gaussian kernel. This is a tensor of type `int32` with shape `(num_targets,)`.
    diameter_to_sigma_factor:
        The factor used to convert the diameter to the standard deviation of the Gaussian kernel. The default value is `6`.
    k_scale:
        The scale applied to the entries within the Gaussian kernel. The default value is `1`.
)doc";

constexpr const char* DOC_DRAW_HEATMAP_BATCHED_IMPL = R"doc(
A function that draws Gaussian heatmaps based on centers and radii.

This function operates on batches and generates one heatmap per sample. Individual samples may have different numbers
of targets (i.e. individual gaussians to draw). However, the inputs are of uniform size, and the individual samples are
padded on the right to match the size of the largest sample. The number of valid target in each sample is given as
an additional input, and only the calid targets are drawn, ignoring the padded elements.

Args:
    heatmaps:
        The heatmaps to be drawn. This is a tensor of type `float32` with shape `(batch_size, height, width)`.
    centers:
        The centers of each Gaussian kernel. This is a tensor of type `int32` with shape `(batch_size, max_num_targets, 2)`,
        where max_num_targets is the maximum number of targets across the batch.
    radii:
        The radii of each Gaussian kernel. This is a tensor of type `int32` with shape `(batch_size, max_num_targets)`,
        where max_num_targets is the maximum number of targets across the batch.
    nums_targets:
        Per-sample number of valid targets. This is a tensor of type `int64` with shape `(batch_size,)`.
    diameter_to_sigma_factor:
        The factor used to convert the diameter to the standard deviation of the Gaussian kernel. The default value is `6`.
    k_scale:
        The scale applied to the entries within the Gaussian kernel. The default value is `1`.
)doc";

constexpr const char* DOC_DRAW_HEATMAP_BATCHED_CLASSWISE_IMPL = R"doc(
A function that draws Gaussian heatmaps based on centers and radii.

This function operates on batches and generates one heatmap for each class. Individual samples may have different numbers
of targets (i.e. individual gaussians to draw). However, the inputs are of uniform size, and the individual samples are
padded on the right to match the size of the largest sample. The number of valid target in each sample is given as
an additional input, and only the calid targets are drawn, ignoring the padded elements.

Args:
    heatmaps:
        The heatmaps to be drawn. This is a tensor of type `float32` with shape `(batch_size, max_num_classes, height, width)`.
        `max_num_classes` is the maximum number of classes in the dataset, e.g. 20 for VOC dataset.
    centers:
        The centers of each Gaussian kernel. This is a tensor of type `int32` with shape `(batch_size, max_num_targets, 2)`,
        where max_num_targets is the maximum number of targets across the batch.
    radii:
        The radii of each Gaussian kernel. This is a tensor of type `int32` with shape `(batch_size, max_num_targets)`,
        where max_num_targets is the maximum number of targets across the batch.
    nums_targets:
        Per-sample number of valid targets. This is a tensor of type `int64` with shape `(batch_size,)`.
    labels:
        The labels of each Gaussian kernel, i.e., bounding box. This is a tensor of type `int32` with shape `(batch_size, max_num_targets)`,
        where max_num_targets is the maximum number of targets across the batch.
        The value corresponds to the class index, e.g. 0 for aeroplane, 1 for bicycle in VOC dataset.
    diameter_to_sigma_factor:
        The factor used to convert the diameter to the standard deviation of the Gaussian kernel. The default value is `6`.
    k_scale:
        The scale applied to the entries within the Gaussian kernel. The default value is `1`.
)doc";

}  // namespace

void draw_heatmap_launcher(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                           const at::Tensor& heatmap_idxes, float diameter_to_sigma_factor, float k_scale);

void draw_heatmap_batched_launcher(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                                   const at::Tensor& nums_targets, float diameter_to_sigma_factor,
                                   float k_scale);

void draw_heatmap_batched_classwise_launcher(at::Tensor& heatmap, const at::Tensor& centers,
                                             const at::Tensor& radii, const at::Tensor& nums_targets,
                                             const at::Tensor& labels, float diameter_to_sigma_factor,
                                             float k_scale);

void draw_heatmap(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                  const at::Tensor& heatmap_idxes, float diameter_to_sigma_factor, float k_scale) {
    return draw_heatmap_launcher(heatmap, centers, radii, heatmap_idxes, diameter_to_sigma_factor, k_scale);
}

void draw_heatmap_batched(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                          const at::Tensor& nums_targets, float diameter_to_sigma_factor, float k_scale) {
    return draw_heatmap_batched_launcher(heatmap, centers, radii, nums_targets, diameter_to_sigma_factor,
                                         k_scale);
}

void draw_heatmap_batched_classwise(at::Tensor& heatmap, const at::Tensor& centers, const at::Tensor& radii,
                                    const at::Tensor& nums_targets, const at::Tensor& labels,
                                    float diameter_to_sigma_factor, float k_scale) {
    return draw_heatmap_batched_classwise_launcher(heatmap, centers, radii, nums_targets, labels,
                                                   diameter_to_sigma_factor, k_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("draw_heatmap", &draw_heatmap, DOC_DRAW_HEATMAP, pybind11::arg("heatmaps"),
          pybind11::arg("centers"), pybind11::arg("radii"), pybind11::arg("heatmap_idxes"),
          pybind11::arg("diameter_to_sigma_factor") = 6.f, pybind11::arg("k_scale") = 1.f);

    m.def("draw_heatmap_batched_impl", &draw_heatmap_batched, DOC_DRAW_HEATMAP_BATCHED_IMPL,
          pybind11::arg("heatmaps"), pybind11::arg("centers"), pybind11::arg("radii"),
          pybind11::arg("nums_targets"), pybind11::arg("diameter_to_sigma_factor") = 6.f,
          pybind11::arg("k_scale") = 1.f);

    m.def("draw_heatmap_batched_classwise_impl", &draw_heatmap_batched_classwise,
          DOC_DRAW_HEATMAP_BATCHED_CLASSWISE_IMPL, pybind11::arg("heatmaps"), pybind11::arg("centers"),
          pybind11::arg("radii"), pybind11::arg("nums_targets"), pybind11::arg("labels"),
          pybind11::arg("diameter_to_sigma_factor") = 6.f, pybind11::arg("k_scale") = 1.f);
}