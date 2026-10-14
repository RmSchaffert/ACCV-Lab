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
from collections.abc import Callable
import csv
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from shapely import points as make_shapely_points
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import plot_frenet_evaluation
import shapely_frenet

DEFAULT_NUM_QUERY_POINTS = [1, 10, 100, 1_000, 10_000]
DEFAULT_NUM_REFERENCE_POINTS = [10, 100]
DEFAULT_NUM_POINTS_PER_MEASUREMENT = 100_000
DEFAULT_NUM_WARMUP_RUNS = 3
DEFAULT_WARMUP_NUM_QUERY_POINTS = 1_000
DEFAULT_SINE_AMPLITUDE = 0.1
DEFAULT_LATERAL_OFFSET = 0.02
DEFAULT_ACCV_PRECISION = "float64"
DEFAULT_ASSERT_ATOL_BY_PRECISION = {
    "float32": 1e-5,
    "float64": 1e-8,
}
DEFAULT_ASSERT_RTOL = 0.0
DEFAULT_OUTPUT_DIR = Path("frenet_eval_results")

DTYPE_NP = np.float64
DTYPE_TORCH_FLOAT64 = torch.float64
DTYPE_TORCH_FLOAT32 = torch.float32


def _torch_dtype_for_precision(accv_precision: str) -> torch.dtype:
    return {
        "float32": DTYPE_TORCH_FLOAT32,
        "float64": DTYPE_TORCH_FLOAT64,
    }[accv_precision]


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _make_reference_lane(num_reference_points: int, sine_amplitude: float) -> np.ndarray:
    if num_reference_points < 2:
        raise ValueError("num_reference_points must be at least 2")

    x = np.linspace(0.0, 2.0 * np.pi, num_reference_points, dtype=DTYPE_NP)
    y = sine_amplitude * np.sin(x)
    return np.column_stack((x, y))


def _make_query_points(
    reference_lane_points: np.ndarray,
    num_query_points: int,
    lateral_offset: float,
) -> np.ndarray:
    segments = np.diff(reference_lane_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    segment_normals = np.stack(
        (-segments[:, 1] / segment_lengths, segments[:, 0] / segment_lengths),
        axis=1,
    )

    sample_indices = np.arange(num_query_points, dtype=np.int64)
    segment_indices = sample_indices % len(segments)
    # Keep projections away from vertices while varying positions within each segment.
    interpolation_factors = 0.25 + 0.5 * ((sample_indices * 0.6180339887498949) % 1.0)
    projection_points = (
        reference_lane_points[segment_indices] + interpolation_factors[:, None] * segments[segment_indices]
    )
    signed_offsets = lateral_offset * np.where(sample_indices % 2 == 0, 1.0, -1.0)
    return projection_points + signed_offsets[:, None] * segment_normals[segment_indices]


def _time_call(
    function: Callable[[], Any],
    *,
    num_runs: int,
    synchronize_cuda: bool,
) -> float:
    if synchronize_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_runs):
        function()
    if synchronize_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start_time) / num_runs


def _validate_results(
    shapely_result: np.ndarray,
    cpu_result: np.ndarray,
    cuda_result: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float]:
    np.testing.assert_allclose(cpu_result, shapely_result, atol=atol, rtol=rtol)
    np.testing.assert_allclose(cuda_result, shapely_result, atol=atol, rtol=rtol)
    np.testing.assert_allclose(cuda_result, cpu_result, atol=atol, rtol=rtol)

    return (
        float(np.max(np.abs(cpu_result - shapely_result))),
        float(np.max(np.abs(cuda_result - shapely_result))),
        float(np.max(np.abs(cuda_result - cpu_result))),
    )


def _run_configuration(
    reference_lane_points: np.ndarray,
    prepared_shapely_reference: shapely_frenet.ShapelyFrenetReference,
    *,
    num_query_points: int,
    lateral_offset: float,
    num_runs: int,
    accv_precision: str,
    accv_dtype: torch.dtype,
    assert_atol: float,
    assert_rtol: float,
) -> dict[str, float | int | str]:
    from accvlab.lane_helpers import frenet

    query_points = _make_query_points(reference_lane_points, num_query_points, lateral_offset)
    shapely_points = make_shapely_points(query_points)

    reference_cpu = torch.from_numpy(reference_lane_points).to(dtype=accv_dtype)
    query_cpu = torch.from_numpy(query_points).to(dtype=accv_dtype)
    reference_cuda = reference_cpu.cuda()
    query_cuda = query_cpu.cuda()

    shapely_call = lambda: shapely_frenet.transform(
        prepared_shapely_reference,
        query_points,
        shapely_points=shapely_points,
    )
    shapely_unprepared_call = lambda: shapely_frenet.transform_unprepared(
        reference_lane_points,
        query_points,
        shapely_points=shapely_points,
    )
    cpu_call = lambda: frenet.transform(reference_cpu, query_cpu)
    cuda_call = lambda: frenet.transform(reference_cuda, query_cuda)

    shapely_result = shapely_call()
    shapely_unprepared_result = shapely_unprepared_call()
    np.testing.assert_allclose(
        shapely_unprepared_result,
        shapely_result,
        atol=0.0,
        rtol=0.0,
    )
    cpu_result = cpu_call().numpy()
    cuda_result = cuda_call().cpu().numpy()
    max_abs_diff_cpu, max_abs_diff_cuda, max_abs_diff_cuda_vs_cpu = _validate_results(
        shapely_result,
        cpu_result,
        cuda_result,
        atol=assert_atol,
        rtol=assert_rtol,
    )

    runtime_shapely_ms = (
        _time_call(
            shapely_call,
            num_runs=num_runs,
            synchronize_cuda=False,
        )
        * 1_000.0
    )
    runtime_shapely_unprepared_ms = (
        _time_call(
            shapely_unprepared_call,
            num_runs=num_runs,
            synchronize_cuda=False,
        )
        * 1_000.0
    )
    runtime_cpu_ms = (
        _time_call(
            cpu_call,
            num_runs=num_runs,
            synchronize_cuda=False,
        )
        * 1_000.0
    )
    runtime_cuda_ms = (
        _time_call(
            cuda_call,
            num_runs=num_runs,
            synchronize_cuda=True,
        )
        * 1_000.0
    )
    return {
        "num_reference_points": len(reference_lane_points),
        "num_query_points": num_query_points,
        "num_runs": num_runs,
        "accv_precision": accv_precision,
        "runtime_shapely_ms": runtime_shapely_ms,
        "runtime_shapely_unprepared_ms": runtime_shapely_unprepared_ms,
        "runtime_cpu_ms": runtime_cpu_ms,
        "runtime_cuda_ms": runtime_cuda_ms,
        "speedup_cpu_vs_shapely": runtime_shapely_ms / runtime_cpu_ms,
        "speedup_cuda_vs_shapely": runtime_shapely_ms / runtime_cuda_ms,
        "speedup_cuda_vs_cpu": runtime_cpu_ms / runtime_cuda_ms,
        "max_abs_diff_cpu_vs_shapely": max_abs_diff_cpu,
        "max_abs_diff_cuda_vs_shapely": max_abs_diff_cuda,
        "max_abs_diff_cuda_vs_cpu": max_abs_diff_cuda_vs_cpu,
    }


def _run_warmup(
    reference_lane_points: np.ndarray,
    prepared_shapely_reference: shapely_frenet.ShapelyFrenetReference,
    *,
    num_query_points: int,
    lateral_offset: float,
    num_runs: int,
    accv_dtype: torch.dtype,
) -> None:
    from accvlab.lane_helpers import frenet

    query_points = _make_query_points(reference_lane_points, num_query_points, lateral_offset)
    shapely_points = make_shapely_points(query_points)
    reference_cpu = torch.from_numpy(reference_lane_points).to(dtype=accv_dtype)
    query_cpu = torch.from_numpy(query_points).to(dtype=accv_dtype)
    reference_cuda = reference_cpu.cuda()
    query_cuda = query_cpu.cuda()

    for _ in range(num_runs):
        shapely_frenet.transform(
            prepared_shapely_reference,
            query_points,
            shapely_points=shapely_points,
        )
        shapely_frenet.transform_unprepared(
            reference_lane_points,
            query_points,
            shapely_points=shapely_points,
        )
        frenet.transform(reference_cpu, query_cpu)
        frenet.transform(reference_cuda, query_cuda)
    torch.cuda.synchronize()


def _write_csv(output_path: Path, results: list[dict[str, float | int | str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def _print_result(result: dict[str, float | int | str]) -> None:
    print(
        f"reference={result['num_reference_points']:>3}, "
        f"queries={result['num_query_points']:>6}, runs={result['num_runs']:>6}: "
        f"Shapely prepared={result['runtime_shapely_ms']:.6f} ms, "
        f"Shapely unprepared={result['runtime_shapely_unprepared_ms']:.6f} ms, "
        f"CPU {result['accv_precision']}={result['runtime_cpu_ms']:.6f} ms, "
        f"CUDA {result['accv_precision']}={result['runtime_cuda_ms']:.6f} ms"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Frenet transformation runtime over transformed-point counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-query-points",
        type=_parse_int_list,
        default=DEFAULT_NUM_QUERY_POINTS,
        help="Comma-separated transformed-point counts.",
    )
    parser.add_argument(
        "--num-reference-points",
        type=_parse_int_list,
        default=DEFAULT_NUM_REFERENCE_POINTS,
        help="Comma-separated point counts for the sine reference lanes.",
    )
    parser.add_argument(
        "--num-points-per-measurement",
        type=int,
        default=DEFAULT_NUM_POINTS_PER_MEASUREMENT,
        help="Target number of transformed points measured per configuration.",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=DEFAULT_NUM_WARMUP_RUNS,
        help="Number of untimed warmup runs.",
    )
    parser.add_argument(
        "--warmup-num-query-points",
        type=int,
        default=DEFAULT_WARMUP_NUM_QUERY_POINTS,
        help="Transformed-point count used during warmup.",
    )
    parser.add_argument(
        "--sine-amplitude",
        type=float,
        default=DEFAULT_SINE_AMPLITUDE,
        help="Amplitude of the smooth sine reference lane.",
    )
    parser.add_argument(
        "--lateral-offset",
        type=float,
        default=DEFAULT_LATERAL_OFFSET,
        help="Absolute normal offset of generated query points.",
    )
    parser.add_argument(
        "--accv-precision",
        choices=tuple(DEFAULT_ASSERT_ATOL_BY_PRECISION),
        default=DEFAULT_ACCV_PRECISION,
        help="Floating-point precision used by the ACCV-Lab CPU and CUDA implementations.",
    )
    parser.add_argument(
        "--assert-atol",
        type=float,
        default=None,
        help="Absolute validation tolerance; defaults to a precision-appropriate value.",
    )
    parser.add_argument(
        "--assert-rtol",
        type=float,
        default=DEFAULT_ASSERT_RTOL,
        help="Relative validation tolerance.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="CSV output path; defaults to frenet_eval_results/point_count_results_<precision>.csv.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Plot output path; defaults to frenet_eval_results/runtime_comparison_<precision>.png.",
    )
    args = parser.parse_args()
    if args.assert_atol is None:
        args.assert_atol = DEFAULT_ASSERT_ATOL_BY_PRECISION[args.accv_precision]
    if args.output_path is None:
        args.output_path = DEFAULT_OUTPUT_DIR / f"point_count_results_{args.accv_precision}.csv"
    if args.plot_path is None:
        args.plot_path = DEFAULT_OUTPUT_DIR / f"runtime_comparison_{args.accv_precision}.png"
    return args


def main() -> None:
    args = _parse_args()
    accv_dtype = _torch_dtype_for_precision(args.accv_precision)
    if not args.output_path.is_absolute():
        args.output_path = SCRIPT_DIR / args.output_path
    if not args.plot_path.is_absolute():
        args.plot_path = SCRIPT_DIR / args.plot_path

    print("Performing Frenet runtime evaluation...")
    print(f"Reference point counts: {args.num_reference_points}")
    print(f"Query point counts: {args.num_query_points}")
    print(f"ACCV-Lab precision: {args.accv_precision}")
    print(f"Output path: {args.output_path}")
    print(f"Plot path: {args.plot_path}")

    results = []
    for num_reference_points in args.num_reference_points:
        reference_lane_points = _make_reference_lane(
            num_reference_points,
            args.sine_amplitude,
        )
        prepared_shapely_reference = shapely_frenet.prepare_reference_lane(reference_lane_points)
        _run_warmup(
            reference_lane_points,
            prepared_shapely_reference,
            num_query_points=args.warmup_num_query_points,
            lateral_offset=args.lateral_offset,
            num_runs=args.num_warmup_runs,
            accv_dtype=accv_dtype,
        )

        for num_query_points in args.num_query_points:
            num_runs = max(1, args.num_points_per_measurement // num_query_points)
            result = _run_configuration(
                reference_lane_points,
                prepared_shapely_reference,
                num_query_points=num_query_points,
                lateral_offset=args.lateral_offset,
                num_runs=num_runs,
                accv_precision=args.accv_precision,
                accv_dtype=accv_dtype,
                assert_atol=args.assert_atol,
                assert_rtol=args.assert_rtol,
            )
            results.append(result)
            _print_result(result)

    _write_csv(args.output_path, results)
    print(f"Wrote {len(results)} result row(s) to {args.output_path}")
    plot_frenet_evaluation.plot_from_csv(
        input_path=args.output_path,
        output_path=args.plot_path,
        reference_point_counts=args.num_reference_points,
    )
    print(f"Wrote runtime comparison plot to {args.plot_path}")


if __name__ == "__main__":
    main()
