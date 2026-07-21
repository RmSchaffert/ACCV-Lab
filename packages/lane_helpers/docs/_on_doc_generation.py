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

from pathlib import Path
import sys
from typing import Any

_POLYLINE_RESULTS_SUBDIR = Path("evaluation_results") / "polyline_runtime_evaluation"
_POLYLINE_GENERATED_IMAGE_SUBDIR = Path("polyline_runtime_evaluation")
_DOC_BATCH_SIZES = [1, 64]
_DOC_FLOAT64_BATCH_SIZES = _DOC_BATCH_SIZES
_DOC_REQUIRED_MARKDOWN_METRICS = (
    "runtime_shapely",
    "runtime_cpu",
    "runtime_cuda",
    "speedup_cpu_vs_shapely",
    "speedup_cuda_vs_shapely",
    "speedup_cuda_vs_cpu",
)
_DOC_REQUIRED_FLOAT64_MARKDOWN_METRICS = (
    "runtime_cpu_float64",
    "runtime_cuda_float64",
    "speedup_cpu_float64_vs_shapely",
    "speedup_cuda_float64_vs_shapely",
    "speedup_cuda_float64_vs_cpu_float64",
)
_DOC_REQUIRED_IMAGE_NAMES = tuple(
    f"batch_{batch_size}_{plot_kind}_comparison.png"
    for batch_size in _DOC_BATCH_SIZES
    for plot_kind in ("runtime", "speedup")
) + tuple(
    f"batch_{batch_size}_{plot_kind}_float64_comparison.png"
    for batch_size in _DOC_FLOAT64_BATCH_SIZES
    for plot_kind in ("runtime", "speedup")
)
_FRENET_RESULTS_PATH = Path("evaluation_results") / "frenet_runtime_evaluation" / "point_count_results.csv"
_FRENET_GENERATED_IMAGE_SUBDIR = Path("frenet_runtime_evaluation")
_FRENET_IMAGE_NAME = "runtime_comparison.png"
_FRENET_REFERENCE_POINT_COUNTS = [10, 100]


def _required_markdown_paths(input_dir: Path) -> list[Path]:
    primary_paths = [
        input_dir / f"batch_{batch_size}_{metric_name}.md"
        for batch_size in _DOC_BATCH_SIZES
        for metric_name in _DOC_REQUIRED_MARKDOWN_METRICS
    ]
    float64_paths = [
        input_dir / f"batch_{batch_size}_{metric_name}.md"
        for batch_size in _DOC_FLOAT64_BATCH_SIZES
        for metric_name in _DOC_REQUIRED_FLOAT64_MARKDOWN_METRICS
    ]
    return primary_paths + float64_paths


def _validate_required_markdown_inputs(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            "Required committed Markdown input directory is missing for lane_helpers docs asset generation: "
            f"{input_dir}."
        )

    missing_inputs = [path for path in _required_markdown_paths(input_dir) if not path.exists()]
    if missing_inputs:
        missing_list = "\n".join(f"  - {path}" for path in missing_inputs)
        raise FileNotFoundError(
            "Missing required committed Markdown input file(s) for lane_helpers docs asset generation:\n"
            f"{missing_list}"
        )


def _validate_required_images(output_dir: Path) -> None:
    missing_outputs = [
        output_dir / image_name
        for image_name in _DOC_REQUIRED_IMAGE_NAMES
        if not (output_dir / image_name).exists()
    ]
    if missing_outputs:
        missing_list = "\n".join(f"  - {path}" for path in missing_outputs)
        raise FileNotFoundError(
            "Polyline runtime docs asset generation did not produce all images referenced by "
            "evaluation_polyline_interpolation.rst:\n"
            f"{missing_list}"
        )


def generate_docs_assets(context: Any) -> None:
    input_dir = context.package_root / _POLYLINE_RESULTS_SUBDIR
    output_dir = context.generated_dir / _POLYLINE_GENERATED_IMAGE_SUBDIR

    _validate_required_markdown_inputs(input_dir)

    evaluation_dir = context.package_root / "evaluation" / "polyline_interpolation"
    sys.path.insert(0, str(evaluation_dir))
    import plot_shapely_evaluation

    plot_shapely_evaluation.plot_from_markdown_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_sizes=_DOC_BATCH_SIZES,
        annotate_plots=True,
    )
    _validate_required_images(output_dir)

    frenet_input_path = context.package_root / _FRENET_RESULTS_PATH
    if not frenet_input_path.exists():
        raise FileNotFoundError(f"Required committed Frenet runtime CSV is missing: {frenet_input_path}.")

    frenet_evaluation_dir = context.package_root / "evaluation" / "frenet"
    sys.path.insert(0, str(frenet_evaluation_dir))
    import plot_frenet_evaluation

    frenet_output_path = context.generated_dir / _FRENET_GENERATED_IMAGE_SUBDIR / _FRENET_IMAGE_NAME
    plot_frenet_evaluation.plot_from_csv(
        input_path=frenet_input_path,
        output_path=frenet_output_path,
        reference_point_counts=_FRENET_REFERENCE_POINT_COUNTS,
    )
    if not frenet_output_path.exists():
        raise FileNotFoundError(
            "Frenet runtime docs asset generation did not produce the image "
            f"referenced by evaluation_frenet.rst: {frenet_output_path}."
        )
