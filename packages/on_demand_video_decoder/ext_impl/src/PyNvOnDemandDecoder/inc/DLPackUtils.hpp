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

#ifndef DLPACKUTILS_HPP
#define DLPACKUTILS_HPP

#include <pybind11/buffer_info.h>
#include <dlpack/dlpack.h>

namespace py = pybind11;

class DLPackTensor final {
   public:
    DLPackTensor() noexcept;
    explicit DLPackTensor(const DLTensor& tensor);
    explicit DLPackTensor(DLManagedTensor&& tensor);
    explicit DLPackTensor(const py::buffer_info& info, const DLDevice& dev);

    DLPackTensor(DLPackTensor&& that) noexcept;
    ~DLPackTensor();

    DLPackTensor& operator=(DLPackTensor&& that) noexcept;

    const DLTensor* operator->() const;
    DLTensor* operator->();

    const DLTensor& operator*() const;
    DLTensor& operator*();

   private:
    DLManagedTensor m_tensor;
};

bool IsCudaAccessible(DLDeviceType devType);

#endif  // DLPACKUTILS_HPP
