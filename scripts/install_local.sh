#!/bin/bash

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

# Convenience wrapper for package_manager.sh install mode.
# This script performs a single, default installation of all namespace packages
# with optional dependencies enabled. It does not accept any parameters.
# For any other installation mode (e.g., without optional dependencies, editable
# installs, wheel building, etc.), please use scripts/package_manager.sh directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -ne 0 ]]; then
    echo "Error: install_local.sh does not accept parameters."
    echo "       For custom installation options, use scripts/package_manager.sh directly."
    exit 1
fi

"$SCRIPT_DIR/package_manager.sh" install --optional