# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(ACCVLAB_MULTI_TENSOR_COPIER_GTEST_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../external/googletest"
)

if(NOT EXISTS "${ACCVLAB_MULTI_TENSOR_COPIER_GTEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GoogleTest submodule not found. Run: git submodule update --init --recursive")
endif()

add_subdirectory(
    "${ACCVLAB_MULTI_TENSOR_COPIER_GTEST_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest"
    EXCLUDE_FROM_ALL
)

foreach(gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
    if(TARGET ${gtest_target})
        target_compile_options(${gtest_target} PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
endforeach()

set(ACCVLAB_MULTI_TENSOR_COPIER_CSRC_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../accvlab/multi_tensor_copier/csrc"
)

add_library(accvlab_multi_tensor_copier_copy_plan_test_impl STATIC EXCLUDE_FROM_ALL
    "${ACCVLAB_MULTI_TENSOR_COPIER_CSRC_DIR}/copy_plan.cpp"
)

target_compile_options(accvlab_multi_tensor_copier_copy_plan_test_impl PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_include_directories(accvlab_multi_tensor_copier_copy_plan_test_impl PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    "${ACCVLAB_MULTI_TENSOR_COPIER_CSRC_DIR}"
)
target_link_libraries(accvlab_multi_tensor_copier_copy_plan_test_impl PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

add_executable(multi_tensor_copier_copy_plan_test EXCLUDE_FROM_ALL
    copy_plan_test.cpp
)

target_compile_options(multi_tensor_copier_copy_plan_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_include_directories(multi_tensor_copier_copy_plan_test PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    "${ACCVLAB_MULTI_TENSOR_COPIER_CSRC_DIR}"
)
target_link_libraries(multi_tensor_copier_copy_plan_test PRIVATE
    GTest::gtest_main
    accvlab_multi_tensor_copier_copy_plan_test_impl
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

add_test(NAME multi_tensor_copier_copy_plan_test COMMAND multi_tensor_copier_copy_plan_test)
set_tests_properties(multi_tensor_copier_copy_plan_test PROPERTIES
    LABELS "multi_tensor_copier_native"
)

set(ACCVLAB_MULTI_TENSOR_COPIER_CTEST_CONFIG_ARGS)
if(CMAKE_CONFIGURATION_TYPES)
    set(ACCVLAB_MULTI_TENSOR_COPIER_CTEST_CONFIG_ARGS -C $<CONFIG>)
endif()

add_custom_target(accvlab_multi_tensor_copier_run_cpp_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
            --output-on-failure
            ${ACCVLAB_MULTI_TENSOR_COPIER_CTEST_CONFIG_ARGS}
            -L multi_tensor_copier_native
    DEPENDS multi_tensor_copier_copy_plan_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running multi_tensor_copier native tests"
    COMMAND_EXPAND_LISTS
)
