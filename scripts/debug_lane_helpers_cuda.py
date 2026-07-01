#!/usr/bin/env python3
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

"""Collect CUDA debug artifacts for sporadic lane_helpers kernel-image failures."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EXTENSION_MODULES = [
    ("lane_helpers", "accvlab.lane_helpers._polyline_sampling", "lane_helpers_polyline_sampling.so"),
    (
        "batching_helpers",
        "accvlab.batching_helpers.batched_indexing_access_cuda",
        "batching_helpers_batched_indexing_access_cuda.so",
    ),
    ("draw_heatmap", "accvlab.draw_heatmap.draw_heatmap_ext", "draw_heatmap_ext.so"),
]

ENV_KEYS = (
    "CUSTOM_CUDA_ARCHS",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_HOME",
    "CUDA_PATH",
    "PATH",
    "LD_LIBRARY_PATH",
    "ACCVLAB_DEBUG_ARTIFACT_DIR",
    "ACCVLAB_DEBUG_PVC_DIR",
    "CUDA_LAUNCH_BLOCKING",
    "VERBOSE_BUILD",
    "BUILD_NUMBER",
    "JOB_NAME",
    "NODE_NAME",
)


def _artifact_dir() -> Path:
    value = os.environ.get("ACCVLAB_DEBUG_ARTIFACT_DIR")
    if not value:
        raise RuntimeError("ACCVLAB_DEBUG_ARTIFACT_DIR is not set")
    return Path(value)


def _run(cmd: List[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(cmd: List[str]) -> Dict[str, Any]:
    try:
        result = _run(cmd)
    except FileNotFoundError as exc:
        return {"command": cmd, "error": str(exc)}
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _git_sha(repo_root: Path) -> str:
    result = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def _detect_build_config(repo_root: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": None, "cuda_info": None, "cmake_args": None, "compile_flags": None}
    try:
        sys.path.insert(0, str(repo_root / "build_config"))
        from accvlab_build_config import build_cmake_args, detect_cuda_info, get_compile_flags, load_config

        cuda_info = detect_cuda_info()
        config = load_config()
        payload["cuda_info"] = cuda_info
        payload["cmake_args"] = build_cmake_args()
        payload["compile_flags"] = get_compile_flags(config, cuda_info)
    except Exception as exc:  # noqa: BLE001 - debug collector
        payload["error"] = repr(exc)
    return payload


def _torch_runtime() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": None, "devices": []}
    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        payload["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(index)
                payload["devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(capability),
                        "arch": f"{capability[0]}{capability[1]}",
                    }
                )
    except Exception as exc:  # noqa: BLE001 - debug collector
        payload["error"] = repr(exc)
    return payload


def _collect_system(repo_root: Path) -> Dict[str, Any]:
    env_snapshot = {key: os.environ.get(key) for key in ENV_KEYS if os.environ.get(key) is not None}
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "uname": _command_output(["uname", "-a"]),
        "git_sha": _git_sha(repo_root),
        "env": env_snapshot,
        "nvidia_smi_l": _command_output(["nvidia-smi", "-L"]),
        "nvidia_smi_q": _command_output(["nvidia-smi", "-q"]),
        "nvcc_version": _command_output(["nvcc", "--version"]),
        "cmake_version": _command_output(["cmake", "--version"]),
        "build_config": _detect_build_config(repo_root),
        "torch_runtime": _torch_runtime(),
    }


def _inspect_extension(label: str, module_name: str, dest_name: str, out_dir: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "label": label,
        "module_name": module_name,
        "dest_name": dest_name,
        "import_error": None,
        "source_path": None,
        "copied_path": None,
        "sha256": None,
        "file": None,
        "ldd": None,
        "cuobjdump_list_elf": None,
        "readelf_sections": None,
    }
    so_dir = out_dir / "so"
    so_dir.mkdir(parents=True, exist_ok=True)

    try:
        module = __import__(module_name, fromlist=["_"])
        module_path = Path(module.__file__).resolve()
        record["source_path"] = str(module_path)
        copied = so_dir / dest_name
        shutil.copy2(module_path, copied)
        record["copied_path"] = str(copied)
        record["sha256"] = _sha256(copied)
        record["file"] = _command_output(["file", str(copied)])
        record["ldd"] = _command_output(["ldd", str(copied)])
        record["cuobjdump_list_elf"] = _command_output(["cuobjdump", "--list-elf", str(copied)])
        record["readelf_sections"] = _command_output(["readelf", "-S", str(copied)])
    except Exception as exc:  # noqa: BLE001 - debug collector
        record["import_error"] = repr(exc)

    return record


def _copy_skbuild_binaries(repo_root: Path, out_dir: Path) -> List[str]:
    copied: List[str] = []
    pattern = repo_root / "packages" / "lane_helpers" / "_skbuild" / "**" / "_polyline_sampling*.so"
    skbuild_dir = out_dir / "so" / "_skbuild"
    skbuild_dir.mkdir(parents=True, exist_ok=True)
    for source in glob.glob(str(pattern), recursive=True):
        source_path = Path(source)
        dest = skbuild_dir / source_path.name
        shutil.copy2(source_path, dest)
        copied.append(str(dest))
    return copied


def _collect_cmake_artifacts(repo_root: Path, out_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"cmake_cache_matches": [], "compile_command_gencode": []}
    cache_pattern = repo_root / "packages" / "lane_helpers" / "_skbuild" / "**" / "CMakeCache.txt"
    for cache_path in glob.glob(str(cache_pattern), recursive=True):
        text = Path(cache_path).read_text(encoding="utf-8", errors="replace")
        matches = [
            line.strip()
            for line in text.splitlines()
            if "CMAKE_CUDA_ARCHITECTURES" in line or "CUDA_ARCHITECTURES" in line
        ]
        payload["cmake_cache_matches"].append({"path": cache_path, "matches": matches})

    compile_commands_pattern = repo_root / "packages" / "lane_helpers" / "_skbuild" / "**" / "compile_commands.json"
    for commands_path in glob.glob(str(compile_commands_pattern), recursive=True):
        text = Path(commands_path).read_text(encoding="utf-8", errors="replace")
        payload["compile_command_gencode"].append(
            {
                "path": commands_path,
                "gencode_sm": sorted(set(re.findall(r"arch=compute_[0-9]+,code=sm_[0-9]+", text))),
                "gencode_ptx": sorted(set(re.findall(r"arch=compute_[0-9]+,code=compute_[0-9]+", text))),
            }
        )

    setup_json = out_dir / "lane_helpers-setup.json"
    if setup_json.is_file():
        payload["lane_helpers_setup"] = json.loads(setup_json.read_text(encoding="utf-8"))

    cmake_config = out_dir / "lane_helpers-cmake-config.txt"
    if cmake_config.is_file():
        payload["lane_helpers_cmake_config"] = cmake_config.read_text(encoding="utf-8", errors="replace")

    return payload


def _grep_install_log(out_dir: Path, patterns: List[str], *, limit: int = 50) -> List[str]:
    install_log = out_dir / "install.log"
    if not install_log.is_file():
        return []
    matches = [
        line
        for line in install_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(pattern in line for pattern in patterns)
    ]
    if len(matches) <= limit:
        return matches
    return matches[-limit:]


def print_console_summary(repo_root: Path) -> None:
    """Print key CUDA/arch diagnostics to stdout for Jenkins console logs."""
    out_dir = _artifact_dir()
    del repo_root  # summary reads from artifact dir only

    print("")
    print("=" * 72)
    print("LANE_HELPERS CUDA DEBUG CONSOLE SUMMARY")
    print("=" * 72)

    system_path = out_dir / "pre_install" / "system.json"
    if system_path.is_file():
        system = json.loads(system_path.read_text(encoding="utf-8"))
        print("\n--- Build-time environment (pre_install) ---")
        for key, label in (
            ("nvidia_smi_l", "nvidia-smi -L"),
            ("nvcc_version", "nvcc"),
            ("cmake_version", "cmake"),
        ):
            payload = system.get(key) or {}
            stdout = (payload.get("stdout") or "").strip()
            if stdout:
                for line in stdout.splitlines()[:3]:
                    print(f"  {label}: {line}")
            elif payload.get("error"):
                print(f"  {label}: error={payload['error']}")

        build_cfg = system.get("build_config") or {}
        print(f"  detect_cuda_info (pre_install): {build_cfg.get('cuda_info')}")
        cmake_args = build_cfg.get("cmake_args") or []
        print("  build_cmake_args():")
        if cmake_args:
            for arg in cmake_args:
                print(f"    {arg}")
        else:
            print("    (empty)")

        compile_flags = build_cfg.get("compile_flags") or {}
        nvcc_flags = [flag for flag in (compile_flags.get("nvcc") or []) if "gencode" in flag]
        print("  get_compile_flags() nvcc gencode (pre_install):")
        if nvcc_flags:
            for flag in nvcc_flags:
                print(f"    {flag}")
        else:
            print("    (none)")

    setup_json = out_dir / "lane_helpers-setup.json"
    if setup_json.is_file():
        print("\n--- lane_helpers setup.py (wheel build) ---")
        payload = json.loads(setup_json.read_text(encoding="utf-8"))
        print(f"  cwd: {payload.get('cwd')}")
        print(f"  cuda_info: {payload.get('cuda_info')}")
        print("  cmake_args:")
        for arg in payload.get("cmake_args") or []:
            print(f"    {arg}")

    cmake_config = out_dir / "lane_helpers-cmake-config.txt"
    if cmake_config.is_file():
        print("\n--- lane_helpers CMake configure ---")
        for line in cmake_config.read_text(encoding="utf-8", errors="replace").strip().splitlines():
            print(f"  {line}")

    install_matches = _grep_install_log(
        out_dir,
        [
            "[lane_helpers debug]",
            "lane_helpers debug:",
            "CMAKE_CUDA_ARCHITECTURES",
            "arch=compute_",
            "code=sm_",
            "code=compute_",
        ],
    )
    if install_matches:
        print("\n--- install.log (lane_helpers / CUDA arch lines) ---")
        for line in install_matches:
            print(f"  {line}")

    post_path = out_dir / "post_install" / "extensions.json"
    if post_path.is_file():
        post = json.loads(post_path.read_text(encoding="utf-8"))
        runtime = post.get("torch_runtime") or {}
        print("\n--- Runtime GPU (post_install) ---")
        devices = runtime.get("devices") or []
        if devices:
            for device in devices:
                print(
                    f"  GPU {device['index']}: {device['name']} "
                    f"capability={device['capability']} arch={device['arch']}"
                )
        else:
            print(
                "  cuda_available="
                f"{runtime.get('cuda_available')} device_count={runtime.get('device_count')} "
                f"error={runtime.get('error')}"
            )

        cmake_art = post.get("cmake_artifacts") or {}
        for entry in cmake_art.get("cmake_cache_matches") or []:
            print(f"\n--- CMakeCache ({entry.get('path')}) ---")
            for line in entry.get("matches") or []:
                print(f"  {line}")

        for entry in cmake_art.get("compile_command_gencode") or []:
            print(f"\n--- compile_commands ({entry.get('path')}) ---")
            for line in entry.get("gencode_sm") or entry.get("gencode") or []:
                print(f"  sm: {line}")
            for line in entry.get("gencode_ptx") or []:
                print(f"  ptx: {line}")

        print("\n--- Extension cuobjdump --list-elf ---")
        for ext in post.get("extensions") or []:
            print(f"  [{ext.get('label')}]")
            cuobjdump = ext.get("cuobjdump_list_elf") or {}
            stdout = (cuobjdump.get("stdout") or "").strip()
            if stdout:
                for line in stdout.splitlines():
                    print(f"    {line}")
            else:
                print(f"    (no output; import_error={ext.get('import_error')})")

    failure = out_dir / "failure_summary.txt"
    if failure.is_file():
        print("\n--- failure_summary.txt ---")
        print(failure.read_text(encoding="utf-8", errors="replace").rstrip())

    print("")
    print("=" * 72)
    print(f"Full artifacts: {out_dir}")
    print("=" * 72)
    print("")


def collect_pre_install(repo_root: Path) -> None:
    out_dir = _artifact_dir()
    payload = _collect_system(repo_root)
    _write_json(out_dir / "pre_install" / "system.json", payload)
    print(f"[debug] wrote {out_dir / 'pre_install' / 'system.json'}")
    print_console_summary(repo_root)


def collect_post_install(repo_root: Path) -> None:
    out_dir = _artifact_dir()
    extensions = [
        _inspect_extension(label, module_name, dest_name, out_dir)
        for label, module_name, dest_name in EXTENSION_MODULES
    ]
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "extensions": extensions,
        "skbuild_copies": _copy_skbuild_binaries(repo_root, out_dir),
        "cmake_artifacts": _collect_cmake_artifacts(repo_root, out_dir),
        "torch_runtime": _torch_runtime(),
    }
    _write_json(out_dir / "post_install" / "extensions.json", payload)
    print(f"[debug] wrote {out_dir / 'post_install' / 'extensions.json'}")
    print_console_summary(repo_root)


def _run_probe(repo_root: Path) -> Dict[str, Any]:
    try:
        from accvlab.lane_helpers.polyline._debug_probe import run_probe

        return run_probe()
    except Exception as exc:  # noqa: BLE001 - debug collector
        return {"error": repr(exc)}


def _write_failure_summary(out_dir: Path, post_install: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("lane_helpers CUDA debug failure summary")
    lines.append("=" * 40)
    if runtime.get("devices"):
        for device in runtime["devices"]:
            lines.append(
                f"runtime GPU {device['index']}: {device['name']} "
                f"capability={device['capability']} arch={device['arch']}"
            )

    setup_json = out_dir / "lane_helpers-setup.json"
    if setup_json.is_file():
        setup_payload = json.loads(setup_json.read_text(encoding="utf-8"))
        lines.append("")
        lines.append("[lane_helpers setup.py]")
        lines.append(f"  cuda_info: {setup_payload.get('cuda_info')}")
        for arg in setup_payload.get("cmake_args") or []:
            lines.append(f"  cmake_arg: {arg}")

    cmake_config = out_dir / "lane_helpers-cmake-config.txt"
    if cmake_config.is_file():
        lines.append("")
        lines.append("[lane_helpers CMake configure]")
        for line in cmake_config.read_text(encoding="utf-8", errors="replace").strip().splitlines():
            lines.append(f"  {line}")

    cmake_art = post_install.get("cmake_artifacts") or {}
    for entry in cmake_art.get("cmake_cache_matches") or []:
        lines.append("")
        lines.append(f"[CMakeCache {entry.get('path')}]")
        for line in entry.get("matches") or []:
            lines.append(f"  {line}")
    for entry in cmake_art.get("compile_command_gencode") or []:
        lines.append("")
        lines.append(f"[compile_commands {entry.get('path')}]")
        for line in entry.get("gencode_sm") or entry.get("gencode") or []:
            lines.append(f"  sm: {line}")
        for line in entry.get("gencode_ptx") or []:
            lines.append(f"  ptx: {line}")

    for ext in post_install.get("extensions", []):
        lines.append("")
        lines.append(f"[{ext['label']}] module={ext['module_name']}")
        if ext.get("import_error"):
            lines.append(f"  import_error: {ext['import_error']}")
            continue
        lines.append(f"  source: {ext.get('source_path')}")
        cuobjdump = ext.get("cuobjdump_list_elf") or {}
        stdout = cuobjdump.get("stdout", "").strip()
        if stdout:
            lines.append("  cuobjdump --list-elf:")
            for line in stdout.splitlines():
                lines.append(f"    {line}")
    (out_dir / "failure_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_on_failure(repo_root: Path) -> None:
    out_dir = _artifact_dir()
    runtime = _torch_runtime()
    _write_json(out_dir / "on_failure" / "runtime.json", runtime)
    _write_json(out_dir / "on_failure" / "probe.json", _run_probe(repo_root))

    install_log = out_dir / "install.log"
    if not install_log.is_file():
        fallback = repo_root / "install.log"
        if fallback.is_file():
            shutil.copy2(fallback, install_log)

    skbuild_src = repo_root / "packages" / "lane_helpers" / "_skbuild"
    skbuild_dest = out_dir / "skbuild_copy"
    if skbuild_src.is_dir():
        if skbuild_dest.exists():
            shutil.rmtree(skbuild_dest)
        shutil.copytree(skbuild_src, skbuild_dest)

    post_install_path = out_dir / "post_install" / "extensions.json"
    post_install: Dict[str, Any] = {}
    if post_install_path.is_file():
        post_install = json.loads(post_install_path.read_text(encoding="utf-8"))
    _write_failure_summary(out_dir, post_install, runtime)
    print(f"[debug] wrote {out_dir / 'failure_summary.txt'}")
    print_console_summary(repo_root)


def bundle(repo_root: Path) -> None:
    out_dir = _artifact_dir()
    manifest = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(repo_root),
        "hostname": platform.node(),
        "build_number": os.environ.get("BUILD_NUMBER"),
        "job_name": os.environ.get("JOB_NAME"),
        "artifact_dir": str(out_dir),
    }
    _write_json(out_dir / "manifest.json", manifest)

    tarball = out_dir / "lane_helpers_cuda_debug.tar.gz"
    if tarball.exists():
        tarball.unlink()
    _run(
        [
            "tar",
            "-czf",
            str(tarball),
            "--exclude",
            tarball.name,
            "-C",
            str(out_dir),
            ".",
        ]
    )

    pvc_dir = os.environ.get("ACCVLAB_DEBUG_PVC_DIR")
    if pvc_dir:
        pvc_path = Path(pvc_dir)
        pvc_path.mkdir(parents=True, exist_ok=True)
        if pvc_path.exists():
            for child in pvc_path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        for child in out_dir.iterdir():
            dest = pvc_path / child.name
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
        shutil.copy2(tarball, pvc_path / tarball.name)
        latest_root = Path(os.environ.get("SHARED_RUN_DIR", pvc_path.parent.parent)) / "debug-artifacts"
        latest_root.mkdir(parents=True, exist_ok=True)
        (latest_root / "LATEST").write_text(str(pvc_path) + "\n", encoding="utf-8")

    print(f"[debug] bundle: {tarball}")
    if pvc_dir:
        print(f"[debug] pvc copy: {pvc_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("pre-install", "post-install", "on-failure", "bundle", "print-summary"),
        help="Collection phase to run",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="ACCV-Lab repository root (defaults to parent of scripts/)",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else script_dir.parent

    if args.phase == "pre-install":
        collect_pre_install(repo_root)
    elif args.phase == "post-install":
        collect_post_install(repo_root)
    elif args.phase == "on-failure":
        collect_on_failure(repo_root)
    elif args.phase == "bundle":
        bundle(repo_root)
    elif args.phase == "print-summary":
        print_console_summary(repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
