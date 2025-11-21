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

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "PyCAIMemoryView.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;
#include "DLPackUtils.hpp"

static std::string ProcessBufferInfoFormat(const std::string& fmt) {
    // pybind11 (as of v2.6.2) doesn't recognize formats 'l' and 'L',
    // which according to https://docs.python.org/3/library/struct.html#format-characters
    // are equal to 'i' and 'I', respectively.
    if (fmt == "l") {
        return "i";
    } else if (fmt == "L") {
        return "I";
    } else {
        return fmt;
    }
}

py::dtype ToDType(const py::buffer_info& info) {
    std::string fmt = ProcessBufferInfoFormat(info.format);

    PyObject* ptr = nullptr;
    if ((py::detail::npy_api::get().PyArray_DescrConverter_(py::str(fmt).ptr(), &ptr) == 0) || !ptr) {
        PyErr_Clear();
        return py::dtype(info);
    } else {
        return py::dtype(fmt);
    }
}

py::dtype ToDType(const std::string& fmt) {
    py::buffer_info buf;
    buf.format = ProcessBufferInfoFormat(fmt);

    return ToDType(buf);
}

DLPackTensor::DLPackTensor() noexcept : m_tensor{} {}

DLPackTensor::DLPackTensor(DLManagedTensor&& managedTensor) : m_tensor{std::move(managedTensor)} {
    managedTensor = {};
}

DLPackTensor::DLPackTensor(const DLTensor& tensor) : DLPackTensor(DLManagedTensor{tensor}) {}

DLPackTensor::DLPackTensor(const py::buffer_info& info, const DLDevice& dev) : m_tensor{} {
    DLTensor& dlTensor = m_tensor.dl_tensor;
    dlTensor.data = info.ptr;

    //TBD dtype

    dlTensor.dtype.code = kDLInt;
    dlTensor.dtype.bits = 8;
    dlTensor.dtype.lanes = 1;

    dlTensor.ndim = info.ndim;
    dlTensor.device = dev;
    dlTensor.byte_offset = 0;

    m_tensor.deleter = [](DLManagedTensor* self) {
        delete[] self->dl_tensor.shape;
        self->dl_tensor.shape = nullptr;

        delete[] self->dl_tensor.strides;
        self->dl_tensor.strides = nullptr;
    };

    try {
        dlTensor.shape = new int64_t[info.ndim];
        std::copy_n(info.shape.begin(), info.shape.size(), dlTensor.shape);

        dlTensor.strides = new int64_t[info.ndim];
        for (int i = 0; i < info.ndim; ++i) {
            if (info.strides[i] % info.itemsize != 0) {
                throw std::runtime_error("Stride must be a multiple of the element size in bytes");
            }

            dlTensor.strides[i] = info.strides[i] / info.itemsize;
        }
    } catch (...) {
        m_tensor.deleter(&m_tensor);
        throw;
    }
}

DLPackTensor::DLPackTensor(DLPackTensor&& that) noexcept : m_tensor{std::move(that.m_tensor)} {
    that.m_tensor = {};
}

DLPackTensor::~DLPackTensor() {
    if (m_tensor.deleter) {
        m_tensor.deleter(&m_tensor);
    }
}

DLPackTensor& DLPackTensor::operator=(DLPackTensor&& that) noexcept {
    if (this != &that) {
        if (m_tensor.deleter) {
            m_tensor.deleter(&m_tensor);
        }
        m_tensor = std::move(that.m_tensor);

        that.m_tensor = {};
    }
    return *this;
}

const DLTensor* DLPackTensor::operator->() const { return &m_tensor.dl_tensor; }

DLTensor* DLPackTensor::operator->() { return &m_tensor.dl_tensor; }

const DLTensor& DLPackTensor::operator*() const { return m_tensor.dl_tensor; }

DLTensor& DLPackTensor::operator*() { return m_tensor.dl_tensor; }

bool IsCudaAccessible(DLDeviceType devType) {
    switch (devType) {
        case kDLCUDAHost:
        case kDLCUDA:
        case kDLCUDAManaged:
            return true;
        default:
            return false;
    }
}
