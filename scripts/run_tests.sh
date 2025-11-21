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

# Run pytest suites for all configured ACCV-Lab namespace packages.
# - Assumes packages are already installed in the current environment.
# - Runs tests from each package's tests/ directory to avoid importing local sources.
# - Warns if a package has no tests; does not treat as error.
# - Exits non-zero if any test run fails.

set -u

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

show_usage() {
    echo "Usage: $0 [--help] [PYTEST_ARGS...] | $0 [--help] -- [PYTEST_ARGS...]"
    echo ""
    echo "Runs pytest for all namespace packages listed in namespace_packages_config.py."
    echo "Arguments are forwarded to pytest; using -- is optional (args after -- are passed verbatim)."
}

# Parse optional --help and passthrough args after --
PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --)
            shift
            PYTEST_ARGS=("$@")
            break
            ;;
        *)
            # Treat all other args as pytest args (no dedicated script flags needed)
            PYTEST_ARGS+=("$1")
            ;;
    esac
    shift || true
done

echo "====================="
echo "ACCV-Lab Test Runner"
echo "====================="

cd "$PROJECT_ROOT"

# Basic plausibility checks
if [ ! -f "namespace_packages_config.py" ]; then
    echo "Error: namespace_packages_config.py not found. Run from ACCV-Lab repo context."
    exit 1
fi

# Discover configured namespace package names (last path element)
PACKAGE_NAMES=$(python3 -c "from namespace_packages_config import get_package_names; print('\n'.join(get_package_names()))" 2>/dev/null || true)

if [ -z "$PACKAGE_NAMES" ]; then
    echo "Warning: No namespace packages discovered from configuration."
    exit 0
fi

TOTAL=0
RAN=0
SKIPPED_NO_DIR=0
SKIPPED_NO_TESTS=0
FAILED=0

for NAME in $PACKAGE_NAMES; do
    TOTAL=$((TOTAL + 1))
    PACKAGE_DIR="$PROJECT_ROOT/packages/$NAME"
    TESTS_DIR="$PACKAGE_DIR/tests"

    echo ""
    echo "[$TOTAL] Package: $NAME"

    if [ ! -d "$PACKAGE_DIR" ]; then
        echo "  Warning: package directory not found: $PACKAGE_DIR (skipping)"
        SKIPPED_NO_DIR=$((SKIPPED_NO_DIR + 1))
        continue
    fi

    if [ ! -d "$TESTS_DIR" ]; then
        echo "  Warning: no tests directory found: $TESTS_DIR (skipping)"
        SKIPPED_NO_TESTS=$((SKIPPED_NO_TESTS + 1))
        continue
    fi

    # Run pytest from within the tests directory to avoid importing local sources
    # and ensure the installed packages are used instead.
    echo "  Running pytest in: $TESTS_DIR"
    pushd "$TESTS_DIR" > /dev/null
    python3 -m pytest -q "${PYTEST_ARGS[@]}"
    RC=$?
    popd > /dev/null

    if [ $RC -ne 0 ]; then
        echo "  ✗ Tests failed for $NAME (exit code $RC)"
        FAILED=$((FAILED + 1))
    else
        echo "  ✓ Tests passed for $NAME"
    fi
    RAN=$((RAN + 1))
done

echo ""
echo "====================="
echo "Summary"
echo "====================="
echo "Packages discovered: $TOTAL"
echo "Packages tested   : $RAN"
echo "Skipped (no dir)  : $SKIPPED_NO_DIR"
echo "Skipped (no tests): $SKIPPED_NO_TESTS"
echo "Failures          : $FAILED"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0


