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

# Debug CI orchestrator for sporadic lane_helpers CUDA kernel-image failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${ACCVLAB_DEBUG_ARTIFACT_DIR:-}" ]]; then
    export ACCVLAB_DEBUG_ARTIFACT_DIR="$PROJECT_ROOT/debug-artifacts"
fi

mkdir -p "$ACCVLAB_DEBUG_ARTIFACT_DIR"

cd "$PROJECT_ROOT"

echo "===== lane_helpers CUDA debug CI ====="
echo "Artifact dir: $ACCVLAB_DEBUG_ARTIFACT_DIR"
echo "PVC dir: ${ACCVLAB_DEBUG_PVC_DIR:-<unset>}"
echo "Git SHA: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Hostname: $(hostname)"

chmod +x "$SCRIPT_DIR/debug_lane_helpers_cuda.sh"

echo ""
echo "===== Phase: pre-install ====="
"$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect pre-install

echo ""
echo "===== Phase: install (tee to install.log) ====="
set +e
"$SCRIPT_DIR/install_local.sh" 2>&1 | tee "$ACCVLAB_DEBUG_ARTIFACT_DIR/install.log"
INSTALL_RC=${PIPESTATUS[0]}
set -e
if [[ "$INSTALL_RC" -ne 0 ]]; then
    echo "Install failed with exit code $INSTALL_RC"
    "$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect on-failure || true
    "$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect bundle || true
    "$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect print-summary || true
    exit "$INSTALL_RC"
fi

echo ""
echo "===== Phase: post-install ====="
"$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect post-install

run_pytest() {
    local package_name="$1"
    local tests_dir="$2"
    shift 2
    echo ""
    echo "===== pytest: $package_name ====="
    pushd "$tests_dir" > /dev/null
    python3 -m pytest -q "$@"
    local rc=$?
    popd > /dev/null
    return "$rc"
}

OVERALL_RC=0

if ! run_pytest batching_helpers \
    "$PROJECT_ROOT/packages/batching_helpers/tests" \
    test_batched_processing_py.py; then
    echo "batching_helpers CUDA smoke failed"
    OVERALL_RC=1
fi

if ! run_pytest draw_heatmap \
    "$PROJECT_ROOT/packages/draw_heatmap/tests" \
    test_draw_heatmap.py; then
    echo "draw_heatmap CUDA smoke failed"
    OVERALL_RC=1
fi

if ! run_pytest lane_helpers \
    "$PROJECT_ROOT/packages/lane_helpers/tests" \
    test_polyline_fixed_interpolation.py test_polyline_var_size_interpolation.py -k cuda; then
    echo "lane_helpers CUDA tests failed"
    OVERALL_RC=1
fi

if [[ "$OVERALL_RC" -ne 0 ]] || [[ "${ACCVLAB_LANE_HELPERS_CUDA_FAILED:-0}" == "1" ]]; then
    echo ""
    echo "===== Phase: on-failure ====="
    "$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect on-failure || true
fi

echo ""
echo "===== Phase: bundle ====="
"$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect bundle

echo ""
echo "===== Phase: console summary (final) ====="
"$SCRIPT_DIR/debug_lane_helpers_cuda.sh" collect print-summary || true

exit "$OVERALL_RC"
