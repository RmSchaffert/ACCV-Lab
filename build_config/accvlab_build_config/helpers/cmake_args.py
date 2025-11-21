import os
from typing import List


def _parse_bool_env(value: str) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_cpp_standard(value: str) -> str:
    """
    Convert inputs like 'c++17' or '17' to a plain numeric standard '17'
    """
    if value is None:
        return ""
    v = value.strip().lower()
    if v.startswith("c++"):
        v = v.replace("c++", "")
    return v


def _detect_cuda_architectures() -> List[str]:
    """
    Try to detect CUDA architectures from PyTorch if available.
    Returns a list like ['70', '75', '80'] ([] if not detected).
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return []
        arches: List[str] = []
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arch = f"{major}{minor}"
            if arch not in arches:
                arches.append(arch)
        return arches
    except Exception:
        return []


def build_cmake_args_from_env() -> List[str]:
    """
    Build a list of -D CMake arguments from environment variables to harmonize
    build configuration across setuptools, external CMake, and scikit-build flows.
    """
    args: List[str] = []
    # Always export compile_commands.json for tooling/validation
    args.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

    # DEBUG_BUILD -> CMAKE_BUILD_TYPE (default to Release if not set)
    if _parse_bool_env(os.environ.get("DEBUG_BUILD", "")):
        args.append("-DCMAKE_BUILD_TYPE=Debug")
    else:
        args.append("-DCMAKE_BUILD_TYPE=Release")

    # CPP_STANDARD -> CMAKE_CXX_STANDARD
    cpp_std = os.environ.get("CPP_STANDARD")
    if cpp_std:
        norm = _normalize_cpp_standard(cpp_std)
        if norm:
            args.append(f"-DCMAKE_CXX_STANDARD={norm}")
            args.append(f"-DCMAKE_CUDA_STANDARD={norm}")

    # CUSTOM_CUDA_ARCHS -> CMAKE_CUDA_ARCHITECTURES
    custom_archs = os.environ.get("CUSTOM_CUDA_ARCHS")
    if custom_archs:
        # Accept comma or semicolon separated
        norm_archs = custom_archs.replace(",", ";")
        args.append(f'-DCMAKE_CUDA_ARCHITECTURES={norm_archs}')
    else:
        # Attempt auto-detection via torch; if empty, let CMake defaults apply
        detected = _detect_cuda_architectures()
        if detected:
            args.append(f'-DCMAKE_CUDA_ARCHITECTURES={";".join(detected)}')

    # VERBOSE_BUILD -> CMAKE_VERBOSE_MAKEFILE
    if _parse_bool_env(os.environ.get("VERBOSE_BUILD", "")):
        args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

    # Aggregate compiler flags
    optimize_level = os.environ.get("OPTIMIZE_LEVEL")
    use_fast_math = _parse_bool_env(os.environ.get("USE_FAST_MATH", ""))
    enable_profiling = _parse_bool_env(os.environ.get("ENABLE_PROFILING", ""))
    verbose_build = _parse_bool_env(os.environ.get("VERBOSE_BUILD", ""))

    cxx_flags: List[str] = []
    cuda_flags: List[str] = []

    if optimize_level:
        cxx_flags.append(f"-O{optimize_level}")
        cuda_flags.append(f"-O{optimize_level}")

    if use_fast_math:
        cxx_flags.append("-ffast-math")
        cuda_flags.append("--use_fast_math")

    if enable_profiling:
        cxx_flags.append("-pg")
        # For nvcc, pass through to host compiler
        cuda_flags.append("-Xcompiler=-pg")

    if verbose_build:
        # Mirror setuptools helper behavior for nvcc ptxas verbosity
        cuda_flags.append("-Xptxas=-v")

    if cxx_flags:
        args.append(f'-DCMAKE_CXX_FLAGS={" ".join(cxx_flags)}')
    if cuda_flags:
        args.append(f'-DCMAKE_CUDA_FLAGS={" ".join(cuda_flags)}')

    return args


if __name__ == "__main__":
    # Print arguments one per line for easy consumption in bash arrays
    for a in build_cmake_args_from_env():
        print(a)
