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

#pragma once

#include <vector>
#include <stdint.h>
#include <mutex>
#include <cuda.h>
#include "NvEncoder.h"

#define CUDA_DRVAPI_CALL( call )                                                                                                 \
    do                                                                                                                           \
    {                                                                                                                            \
        CUresult err__ = call;                                                                                                   \
        if (err__ != CUDA_SUCCESS)                                                                                               \
        {                                                                                                                        \
            const char *szErrName = NULL;                                                                                        \
            cuGetErrorName(err__, &szErrName);                                                                                   \
            std::ostringstream errorLog;                                                                                         \
            errorLog << "CUDA driver API error " << szErrName ;                                                                  \
            throw NVENCException::makeNVENCException(errorLog.str(), NV_ENC_ERR_GENERIC, __FUNCTION__, __FILE__, __LINE__);      \
        }                                                                                                                        \
    }                                                                                                                            \
    while (0)

/**
*  @brief Encoder for CUDA device memory.
*/
class NvEncoderCuda : public NvEncoder
{
public:
    NvEncoderCuda(CUcontext cuContext, CUstream cuStream,uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
        uint32_t nExtraOutputDelay = 3, bool bMotionEstimationOnly = false, bool bOPInVideoMemory = false, bool bUseIVFContainer = true);
    virtual ~NvEncoderCuda();

    /**
    *  @brief This is a static function to copy input data from host memory to device memory.
    *  This function assumes YUV plane is a single contiguous memory segment.
    */
    static void CopyToDeviceFrame(CUcontext device,
        void* pSrcFrame,
        uint32_t nSrcPitch,
        CUdeviceptr pDstFrame,
        uint32_t dstPitch,
        int width,
        int height,
        CUmemorytype srcMemoryType,
        NV_ENC_BUFFER_FORMAT pixelFormat,
        const uint32_t dstChromaOffsets[],
        uint32_t numChromaPlanes,
        bool bUnAlignedDeviceCopy = false,
        CUstream stream = NULL,
        const uint32_t srcChromaOffsets[] = NULL);

    /**
    *  @brief This is a static function to copy input data from host memory to device memory.
    *  Application must pass a seperate device pointer for each YUV plane.
    */
    static void CopyToDeviceFrame(CUcontext device,
        void* pSrcFrame,
        uint32_t nSrcPitch,
        CUdeviceptr pDstFrame,
        uint32_t dstPitch,
        int width,
        int height,
        CUmemorytype srcMemoryType,
        NV_ENC_BUFFER_FORMAT pixelFormat,
        CUdeviceptr dstChromaPtr[],
        uint32_t dstChromaPitch,
        uint32_t numChromaPlanes,
        bool bUnAlignedDeviceCopy = false);

    NV_ENCODE_API_FUNCTION_LIST GetApi() const { return m_nvenc;}

    void*                       GetEncoder() const { return m_hEncoder;}
    /**
    *  @brief This function sets input and output CUDA streams
    */
    void SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream);

protected:
    /**
    *  @brief This function is used to release the input buffers allocated for encoding.
    *  This function is an override of virtual function NvEncoder::ReleaseInputBuffers().
    */
    virtual void ReleaseInputBuffers() override;

private:
    /**
    *  @brief This function is used to allocate input buffers for encoding.
    *  This function is an override of virtual function NvEncoder::AllocateInputBuffers().
    */
    virtual void AllocateInputBuffers(int32_t numInputBuffers) override;

private:
    /**
    *  @brief This is a private function to release CUDA device memory used for encoding.
    */
    void ReleaseCudaResources();

protected:
    CUcontext m_cuContext;
    CUstream  m_cuStream;

private:
    size_t m_cudaPitch = 0;
};
