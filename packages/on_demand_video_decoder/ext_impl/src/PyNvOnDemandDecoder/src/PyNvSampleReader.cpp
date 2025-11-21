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

#include "PyNvSampleReader.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvtx3/nvtx3.hpp"

PyNvSampleReader::PyNvSampleReader(int num_of_set, int num_of_file, int iGpu,
                                   bool bSuppressNoColorRangeWarning)
    : num_of_set(num_of_set),
      num_of_file(num_of_file),
      gpu_id(iGpu),
      suppress_no_color_range_given_warning(bSuppressNoColorRangeWarning) {
#ifdef IS_DEBUG_BUILD
    std::cout << "New PyNvSampleReader object" << std::endl;
#endif
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]"
                  << std::endl;
    }
    this->destroy_context = false;

    // To do, we can reuse current context, we can check its func
    // ck(cuCtxGetCurrent(&cuContext));
    this->cu_context = nullptr;
    if (!this->cu_context) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, this->gpu_id));
        ck(cuDevicePrimaryCtxRetain(&this->cu_context, cuDevice));
        ck(cuCtxPushCurrent(this->cu_context));
        this->destroy_context = true;
    }
    if (!this->cu_context) {
        throw std::domain_error(
            "[ERROR] Failed to create a cuda context. Create a "
            "cudacontext and pass it as "
            "named argument 'cudacontext = app_ctx'");
    }
    ck(cuStreamCreate(&this->cu_stream, CU_STREAM_DEFAULT));
    VideoReaderMap.reserve(this->num_of_file);
    for (int i = 0; i < this->num_of_file; i++) {
        VideoReaderMap.emplace_back(this->num_of_set);
    }
}

PyNvSampleReader::~PyNvSampleReader() {
#ifdef IS_DEBUG_BUILD
    std::cout << "Delete PyNvSampleReader object" << std::endl;
#endif

    this->clearAllReaders();

    if (this->cu_stream) {
        ck(cuStreamDestroy(this->cu_stream));
    }
    if (this->destroy_context) {
        ck(cuCtxPopCurrent(&this->cu_context));
        ck(cuDevicePrimaryCtxRelease(this->gpu_id));
    }
}

// Clear all video readers before destroying context
void PyNvSampleReader::clearAllReaders() {
    for (auto& reader_map : VideoReaderMap) {
        reader_map.clearAllReaders();
    }
}

// Helper function to process video frames in parallel
template <typename T, typename Func>
std::vector<T> process_frames_in_parallel(const std::vector<std::string>& filepaths,
                                          const std::vector<int>& frame_ids,
                                          const std::vector<PyNvVideoReader*>& video_readers,
                                          Func process_frame) {
    nvtxRangePushA("Process Frames in Parallel");
    std::vector<T> res(filepaths.size());
    std::exception_ptr eptr = nullptr;
    std::mutex mutex;

    std::vector<std::thread> threads;
    threads.reserve(filepaths.size());

    for (int i = 0; i < filepaths.size(); i++) {
        threads.emplace_back([&, i]() {
            try {
                res[i] = process_frame(video_readers[i], frame_ids[i]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mutex);
                eptr = std::current_exception();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (eptr) {
        nvtxRangePop();
        std::rethrow_exception(eptr);
    }
    nvtxRangePop();
    return res;
}

std::vector<RGBFrame> PyNvSampleReader::run_rgb_out(const std::vector<std::string>& filepaths,
                                                    const std::vector<int> frame_ids, bool as_bgr) {
    assert(filepaths.size() == frame_ids.size());
    std::vector<PyNvVideoReader*> video_readers(filepaths.size());

    nvtxRangePushA("Get Video Readers");
    for (int i = 0; i < filepaths.size(); i++) {
        FixedSizeVideoReaderMap& reader_map = this->VideoReaderMap[i];
        PyNvVideoReader* video_reader = nullptr;

        if (reader_map.notFull()) {
            video_reader = new PyNvVideoReader(filepaths[i], this->gpu_id, this->cu_context, this->cu_stream);
        }

        auto cur_video_reader = reader_map.find(filepaths[i], video_reader);
        video_readers[i] = cur_video_reader;
    }
    nvtxRangePop();

    return process_frames_in_parallel<RGBFrame>(filepaths, frame_ids, video_readers,
                                                [as_bgr](PyNvVideoReader* reader, int frame_id) {
                                                    return reader->run_single_rgb_out(frame_id, as_bgr);
                                                });
}

std::vector<DecodedFrameExt> PyNvSampleReader::run(const std::vector<std::string>& filepaths,
                                                   const std::vector<int> frame_ids) {
    assert(filepaths.size() == frame_ids.size());
    std::vector<PyNvVideoReader*> video_readers(filepaths.size());

    for (int i = 0; i < filepaths.size(); i++) {
        FixedSizeVideoReaderMap& reader_map = this->VideoReaderMap[i];
        PyNvVideoReader* video_reader = nullptr;

        if (reader_map.notFull()) {
            video_reader = new PyNvVideoReader(filepaths[i], this->gpu_id, this->cu_context, this->cu_stream);
        }

        auto cur_video_reader = reader_map.find(filepaths[i], video_reader);
        video_readers[i] = cur_video_reader;
    }

    return process_frames_in_parallel<DecodedFrameExt>(
        filepaths, frame_ids, video_readers,
        [](PyNvVideoReader* reader, int frame_id) { return reader->run_single(frame_id); });
}

void Init_PyNvSampleReader(py::module& m) {
    // Create a factory function for convenient instantiation
    m.def(
        "CreateSampleReader",
        [](int num_of_set, int num_of_file, int iGpu, bool suppressNoColorRangeWarning) {
            return std::make_shared<PyNvSampleReader>(num_of_set, num_of_file, iGpu,
                                                      suppressNoColorRangeWarning);
        },
        py::arg("num_of_set"), py::arg("num_of_file"), py::arg("iGpu") = 0,
        py::arg("suppressNoColorRangeWarning") = false,
        R"pbdoc(
            Initialize sample reader with multiple video readers.
            
            This factory function creates a PyNvSampleReader instance with the specified
            configuration for high-throughput multi-file video processing. It's the
            recommended way to create sample reader instances.
            
            Args:
                num_of_set: Number of video readers per file for parallel processing
                num_of_file: Number of files to handle simultaneously
                iGpu: GPU device ID to use for decoding (0 for primary GPU)
                suppressNoColorRangeWarning: Suppress warning when no color range can be extracted from video files (limited/MPEG range is assumed)
            
            Returns:
                PyNvSampleReader instance configured with the specified parameters
            
            Raises:
                RuntimeError: If GPU initialization fails or parameters are invalid
            
            Example:
                >>> reader = CreateSampleReader(num_of_set=2, num_of_file=3, iGpu=0)
                >>> frames = reader.Decode(['v0.mp4', 'v1.mp4'], [0, 10])
            
            Note:
                The parameter `num_of_set` in `nvc.CreateSampleReader` controls the decoding cycle:
                - For a specific decoder instance, if you are decoding clipA, after calling `DecodeN12ToRGB` `num_of_set` times, the input returns to clipA again
                - If you are continuously decoding the same clip, then `num_of_set` can be set to 1
            )pbdoc");

    // Define the PyNvSampleReader class and its methods
    py::class_<PyNvSampleReader, shared_ptr<PyNvSampleReader>>(m, "PyNvSampleReader", py::module_local(),
                                                               R"pbdoc(
        NVIDIA GPU-accelerated sample reader for multi-file video processing.
        
        This class provides high-performance video reading capabilities using NVIDIA
        hardware acceleration for multiple video files with multiple readers per file.
        It's designed for scenarios requiring high-throughput processing of multiple
        video streams simultaneously.
        
        Key Features:
        - GPU-accelerated decoding using NVIDIA hardware
        - Multiple video readers per file for parallel processing
        - Multi-file support with configurable reader pools
        - RGB and YUV output formats
        - Resource management with explicit cleanup
        - Optimized for high-throughput batch processing
        )pbdoc")
        .def(py::init<int, int, int, bool>(), py::arg("num_of_set"), py::arg("num_of_file"),
             py::arg("iGpu") = 0, py::arg("suppressNoColorRangeWarning") = false,
             R"pbdoc(
            Initialize sample reader with set of particular parameters.
            
            Args:
                num_of_set: Number of video readers per file for parallel processing
                num_of_file: Number of files to handle simultaneously
                iGpu: GPU device ID to use for decoding (0 for primary GPU)
                suppressNoColorRangeWarning: Suppress warning when no color range can be extracted from video files (limited/MPEG range is assumed)
            
            Raises:
                RuntimeError: If GPU initialization fails or parameters are invalid

            Note:
                The parameter `num_of_set` in `nvc.CreateSampleReader` controls the decoding cycle:
                - For a specific decoder instance, if you are decoding clipA, after calling `DecodeN12ToRGB` `num_of_set` times, the input returns to clipA again
                - If you are continuously decoding the same clip, then `num_of_set` can be set to 1
            )pbdoc")
        .def(
            "Decode",
            [](std::shared_ptr<PyNvSampleReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids) {
                try {
                    return reader->run(filepaths, frame_ids);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"),
            R"pbdoc(
            Decodes video frames into uncompressed YUV data.
            
            This method performs GPU-accelerated decoding of specific frames from multiple
            video files using the configured reader pools. It returns frames in YUV format
            with metadata.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
            
            Returns:
                List of DecodedFrameExt objects containing the decoded frame data.
                Each frame includes YUV pixel data, metadata, and timing information.
            
            Raises:
                RuntimeError: If video files cannot be decoded or frame IDs are invalid
                ValueError: If frame_ids contain invalid indices or filepaths is empty
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> frames = reader.Decode(['video1.mp4', 'video2.mp4'], [0, 10])
                >>> print(f"Decoded {len(frames)} frames")
            )pbdoc")
        .def(
            "DecodeN12ToRGB",
            [](std::shared_ptr<PyNvSampleReader>& reader, const std::vector<std::string>& filepaths,
               const std::vector<int> frame_ids, bool as_bgr) {
                try {
                    return reader->run_rgb_out(filepaths, frame_ids, as_bgr);
                } catch (const std::exception& e) {
                    throw std::runtime_error(e.what());
                }
            },
            py::arg("filepaths"), py::arg("frame_ids"), py::arg("as_bgr") = false,
            R"pbdoc(
            Decodes video frames and converts them to RGB/BGR format.
            
            This method performs GPU-accelerated decoding and color space conversion
            from YUV to RGB/BGR format for multiple video files. It's optimized for
            machine learning applications that require RGB input data.
            
            Args:
                filepaths: List of video file paths to decode from
                frame_ids: List of frame IDs to decode from the video files
                as_bgr: Whether to output in BGR format (True) or RGB format (False). BGR is commonly used in OpenCV applications.
            
            Returns:
                List of RGBFrame objects containing the decoded and color-converted frame data.
                Each frame includes RGB/BGR pixel data and metadata.
            
            Raises:
                RuntimeError: If video files cannot be decoded or frame IDs are invalid
                ValueError: If frame_ids contain invalid indices or filepaths is empty
            
            Example:

                Ref to Sample: `samples/SampleStreamAccess.py`
                
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> rgb_frames = reader.DecodeN12ToRGB(['video1.mp4', 'video2.mp4'], [0, 10], as_bgr=True)
                >>> print(f"Decoded {len(rgb_frames)} RGB frames")
            )pbdoc")
        .def(
            "clearAllReaders", [](std::shared_ptr<PyNvSampleReader>& reader) { reader->clearAllReaders(); },
            R"pbdoc(
            Clear all video readers and release associated resources.
            
            This method releases all video reader instances and their associated
            GPU resources. It should be called when the reader is no longer needed
            to free up GPU memory and other system resources.
            
            Example:
                >>> reader = PyNvSampleReader(num_of_set=2, num_of_file=3)
                >>> frames = reader.Decode(['video1.mp4'], [0, 10, 20])
                >>> reader.clearAllReaders()  # Clean up resources
            )pbdoc");
}