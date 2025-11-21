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

# Main formatting script for ACCV-Lab
# Usage: ./scripts/format.sh [OPTIONS] [TARGET]
#
# OPTIONS:
#   --python, -p          Format Python code only
#   --cpp, -c             Format C++/CUDA code only
#   --all, -a             Format both Python and C++ (default)
#   --common-only         Format only common code (root files, docs, build_config)
#   --include-packages    Format common code + all namespace packages
#   --package <name>      Format specific namespace package only
#   --help, -h            Show this help message
#
# EXAMPLES:
#   ./scripts/format.sh                           # Format everything
#   ./scripts/format.sh --python                  # Format Python only
#   ./scripts/format.sh --cpp --package examples  # Format C++ in examples package
#   ./scripts/format.sh --common-only             # Format only common code

set -e

# Default values
FORMAT_PYTHON=true
FORMAT_CPP=true
FORMAT_COMMON_ONLY=false
FORMAT_INCLUDE_PACKAGES=false
TARGET_PACKAGE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python|-p)
            FORMAT_PYTHON=true
            FORMAT_CPP=false
            shift
            ;;
        --cpp|-c)
            FORMAT_PYTHON=false
            FORMAT_CPP=true
            shift
            ;;
        --all|-a)
            FORMAT_PYTHON=true
            FORMAT_CPP=true
            shift
            ;;
        --common-only)
            FORMAT_COMMON_ONLY=true
            shift
            ;;
        --include-packages)
            FORMAT_INCLUDE_PACKAGES=true
            shift
            ;;
        --package)
            TARGET_PACKAGE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [TARGET]"
            echo ""
            echo "OPTIONS:"
            echo "  --python, -p          Format Python code only"
            echo "  --cpp, -c             Format C++/CUDA code only"
            echo "  --all, -a             Format both Python and C++ (default)"
            echo "  --common-only         Format only common code"
            echo "  --include-packages    Format common code + all namespace packages"
            echo "  --package <name>      Format specific namespace package only"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "EXAMPLES:"
            echo "  $0                           # Format everything"
            echo "  $0 --python                  # Format Python only"
            echo "  $0 --cpp --package examples  # Format C++ in examples package"
            echo "  $0 --common-only             # Format only common code"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORMATTING_DIR="$SCRIPT_DIR/formatting"

echo "ACCV-Lab Code Formatter"
echo "====================="

# Function to run Python formatting
format_python() {
    if [ "$FORMAT_COMMON_ONLY" = true ]; then
        echo "Formatting Python common code..."
        "$FORMATTING_DIR/black_format_package.sh"
    elif [ -n "$TARGET_PACKAGE" ]; then
        echo "Formatting Python in package: $TARGET_PACKAGE"
        "$FORMATTING_DIR/black_format_subpackage.sh" "$TARGET_PACKAGE"
    elif [ "$FORMAT_INCLUDE_PACKAGES" = true ]; then
        echo "Formatting Python common code and all packages..."
        "$FORMATTING_DIR/black_format_package.sh" --include-subpackages
    else
        echo "Formatting Python common code and all packages..."
        "$FORMATTING_DIR/black_format_package.sh" --include-subpackages
    fi
}

# Function to run C++ formatting
format_cpp() {
    if [ "$FORMAT_COMMON_ONLY" = true ]; then
        echo "No C++ code in common areas, skipping..."
        return
    elif [ -n "$TARGET_PACKAGE" ]; then
        echo "Formatting C++ in package: $TARGET_PACKAGE"
        "$FORMATTING_DIR/clang_format_subpackage.sh" "$TARGET_PACKAGE"
    elif [ "$FORMAT_INCLUDE_PACKAGES" = true ]; then
        echo "Formatting C++ in all packages..."
        "$FORMATTING_DIR/clang_format_package.sh" --include-subpackages
    else
        echo "Formatting C++ in all packages..."
        "$FORMATTING_DIR/clang_format_package.sh" --include-subpackages
    fi
}

# Run formatting based on options
if [ "$FORMAT_PYTHON" = true ]; then
    format_python
fi

if [ "$FORMAT_CPP" = true ]; then
    format_cpp
fi

echo "Formatting completed successfully!" 