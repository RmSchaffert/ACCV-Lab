# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

message(STATUS "Using FFMPEG_DIR=${FFMPEG_DIR}")
set(NV_FFMPEG_LIBRARIES "")

message(STATUS "Using CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")

macro(link_av_component target lib_name)
	find_library(${lib_name}_library
		NAMES ${lib_name}
		HINTS "${FFMPEG_DIR}/lib"
	)
	message(STATUS "Link ${${lib_name}_library}")
	list(APPEND NV_FFMPEG_LIBRARIES ${${lib_name}_library})
endmacro()

link_av_component(VideoCodecSDKUtils avformat)
link_av_component(VideoCodecSDKUtils avcodec)
link_av_component(VideoCodecSDKUtils swresample)
link_av_component(VideoCodecSDKUtils avutil)

find_path(
    BSF_INCLUDE_DIR
    NAMES "libavcodec/bsf.h"
    HINTS ${TC_FFMPEG_INCLUDE_DIR} "${FFMPEG_DIR}/include"
)
if(BSF_INCLUDE_DIR)
    set(NV_FFMPEG_HAS_BSF TRUE)
else()
    set(NV_FFMPEG_HAS_BSF FALSE)
    message(WARNING "Could not find \"libavcodec/bsf.h\" while other ffmpeg includes could be found."
        "This likely means that your FFMPEG installation is old. Still trying to compile VPF!")
endif()
