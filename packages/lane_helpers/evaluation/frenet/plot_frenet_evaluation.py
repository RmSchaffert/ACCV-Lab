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

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _read_results(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader)


def plot_from_csv(
    *,
    input_path: Path,
    output_path: Path,
    reference_point_counts: list[int],
) -> Path:
    """Plot Frenet runtimes for the requested reference-lane sizes."""
    results = _read_results(input_path)
    if not results:
        raise ValueError(f"No benchmark rows found in {input_path}")

    precisions = {result.get("accv_precision", "float64") for result in results}
    if len(precisions) != 1:
        raise ValueError(f"Expected one ACCV-Lab precision in {input_path}, got {sorted(precisions)}")
    runtime_series = (
        ("runtime_shapely_ms", "Shapely prepared", "C0", "-"),
        ("runtime_shapely_unprepared_ms", "Shapely unprepared", "C0", "--"),
        ("runtime_cpu_ms", "ACCV-Lab CPU", "C1", "-"),
        ("runtime_cuda_ms", "ACCV-Lab CUDA", "C2", "-"),
    )

    figure, axes = plt.subplots(
        1,
        len(reference_point_counts),
        figsize=(6.2 * len(reference_point_counts), 4.8),
        sharey=True,
        squeeze=False,
    )
    for axis, num_reference_points in zip(axes[0], reference_point_counts, strict=True):
        matching_results = sorted(
            (result for result in results if int(result["num_reference_points"]) == num_reference_points),
            key=lambda result: int(result["num_query_points"]),
        )
        if not matching_results:
            raise ValueError(f"No benchmark rows found for {num_reference_points} reference points")

        query_point_counts = np.asarray([int(result["num_query_points"]) for result in matching_results])
        for metric_name, label, color, line_style in runtime_series:
            runtimes = np.asarray([float(result[metric_name]) for result in matching_results])
            axis.plot(
                query_point_counts,
                runtimes,
                color=color,
                linestyle=line_style,
                marker="o",
                label=label,
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{num_reference_points} reference points")
        axis.set_xlabel("Number of transformed points")
        axis.grid(True, which="both", alpha=0.3)

    axes[0, 0].set_ylabel("Runtime per call [ms]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Frenet runtime results from CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--num-reference-points",
        type=_parse_int_list,
        default=[10, 100],
        help="Comma-separated reference-lane point counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_from_csv(
        input_path=args.input_path,
        output_path=args.output_path,
        reference_point_counts=args.num_reference_points,
    )


if __name__ == "__main__":
    main()
