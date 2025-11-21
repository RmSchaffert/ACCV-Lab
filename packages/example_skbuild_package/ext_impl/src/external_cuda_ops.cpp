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

#include <torch/extension.h>
#include "../include/external_cuda_ops.h"

namespace {

constexpr const char* DOC_EXTERNAL_VECTOR_ADD_CUDA = R"doc(
Vector addition using external CUDA implementation.

:gpu:

Args:
    a: First input 1D tensor on a CUDA device.
    b: Second input 1D tensor on a CUDA device.

Returns:
    Tensor on CUDA containing the element-wise sum of ``a`` and ``b``.
)doc";

constexpr const char* DOC_EXTERNAL_VECTOR_SCALE_CUDA = R"doc(
Vector scaling using external CUDA implementation.

:gpu:

Args:
    a: Input 1D tensor on a CUDA device.
    scale: Scalar factor used to scale all elements of ``a``.

Returns:
    Tensor on CUDA containing the scaled values of ``a``.
)doc";

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("external_vector_add_cuda", &external_vector_add_cuda, DOC_EXTERNAL_VECTOR_ADD_CUDA);
    m.def("external_vector_scale_cuda", &external_vector_scale_cuda, DOC_EXTERNAL_VECTOR_SCALE_CUDA);
}