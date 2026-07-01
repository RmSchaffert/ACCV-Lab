#!/bin/bash

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

# Thin wrapper around scripts/debug_lane_helpers_cuda.py for CI shell stages.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 collect {pre-install|post-install|on-failure|bundle|print-summary}"
}

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
PHASE="$2"

if [[ "$COMMAND" != "collect" ]]; then
    usage
    exit 1
fi

case "$PHASE" in
    pre-install)
        PHASE_ARG="pre-install"
        ;;
    post-install)
        PHASE_ARG="post-install"
        ;;
    on-failure)
        PHASE_ARG="on-failure"
        ;;
    bundle)
        PHASE_ARG="bundle"
        ;;
    print-summary)
        PHASE_ARG="print-summary"
        ;;
    *)
        usage
        exit 1
        ;;
esac

python3 "$SCRIPT_DIR/debug_lane_helpers_cuda.py" "$PHASE_ARG" --repo-root "$PROJECT_ROOT"
