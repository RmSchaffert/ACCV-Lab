#!/usr/bin/env python3

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

"""
Script to materialize additional directories referenced by documentation from individual namespace packages.
This script reads docu_referenced_dirs.txt from each package and symlinks (or copies) the specified directories
to the contained_package_docs_mirror directory.
"""

import sys
import os
import argparse
import functools
import shutil
from pathlib import Path
from textwrap import dedent


def vprint(*args, verbose=True, **kwargs):
    """Print only if verbose mode is enabled"""
    if verbose:
        print(*args, **kwargs)


def read_referenced_dirs(package_dir):
    """Read the docu_referenced_dirs.txt file from a package directory"""
    referenced_dirs_file = package_dir / "docu_referenced_dirs.txt"

    # Always include docs directory
    referenced_dirs = ["docs"]

    if not referenced_dirs_file.exists():
        # Return only docs if file doesn't exist
        return referenced_dirs

    try:
        with open(referenced_dirs_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Process lines, skipping comments and empty lines
        for line in lines:
            # Skip empty lines and comments (lines starting with #)
            if line and not line.startswith('#'):
                referenced_dirs.append(line)

        return referenced_dirs
    except Exception as e:
        print(f"Warning: Could not read {referenced_dirs_file}: {e}")
        return referenced_dirs


def ensure_removed(path: Path):
    """Remove existing path (file, dir, or symlink) if present."""
    try:
        if path.is_symlink() or path.exists():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    except Exception as e:
        raise RuntimeError(f"Failed to remove existing path {path}: {e}")


def copy_directory(src_dir: Path, dst_dir: Path, cprint, mode: str = "symlink"):
    """Materialize src_dir at dst_dir using specified mode ('symlink' or 'copy')."""
    if not src_dir.exists():
        cprint(f"  Warning: Source directory does not exist: {src_dir}")
        return False

    try:
        ensure_removed(dst_dir)
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            # Create a relative symlink for portability
            rel_target = os.path.relpath(str(src_dir), start=str(dst_dir.parent))
            os.symlink(rel_target, str(dst_dir))
            cprint(f"  Symlinked: {dst_dir} -> {rel_target}")
        else:
            shutil.copytree(src_dir, dst_dir)
            cprint(f"  Copied: {src_dir} -> {dst_dir}")
        return True
    except Exception as e:
        cprint(f"  Error materializing {src_dir} -> {dst_dir} ({mode}): {e}")
        return False


def copy_package_referenced_dirs(package_name, package_dir, target_base_dir, cprint, mode: str = "symlink"):
    """Copy or symlink all referenced directories for a specific package"""
    cprint(f"Processing package: {package_name}")

    # Read the list of directories to copy
    referenced_dirs = read_referenced_dirs(package_dir)
    cprint(f"  Referenced directories: {referenced_dirs}")

    # Create the target package directory
    target_package_dir = target_base_dir / package_name
    target_package_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for dir_name in referenced_dirs:
        src_dir = package_dir / dir_name
        dst_dir = target_package_dir / dir_name

        if copy_directory(src_dir, dst_dir, cprint, mode=mode):
            copied_count += 1

    cprint(f"  Successfully copied {copied_count}/{len(referenced_dirs)} directories")
    return copied_count


def main():
    """Main function to copy referenced directories from all namespace packages"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Copy additional directories referenced by documentation from namespace packages'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument(
        '--mode',
        choices=['symlink', 'copy'],
        default='symlink',
        help='Materialization mode: symlink (default) or copy',
    )
    args = parser.parse_args()

    # Configure conditional print function once
    cprint = functools.partial(vprint, verbose=args.verbose)

    # Get the project root (parent of docs directory)
    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent

    cprint(f"Project root: {project_root}")
    cprint(f"Docs directory: {docs_dir}")

    # Import the namespace packages list from the shared config
    sys.path.insert(0, str(project_root))
    try:
        from namespace_packages_config import NAMESPACE_PACKAGES

        namespace_packages = NAMESPACE_PACKAGES
        cprint(f"Using configured namespace packages: {namespace_packages}")
    except ImportError as e:
        print(f"Error: Could not import NAMESPACE_PACKAGES from namespace_packages_config.py: {e}")
        return 1
    except AttributeError:
        print("Error: NAMESPACE_PACKAGES not found in namespace_packages_config.py")
        return 1

    if not namespace_packages:
        print("Warning: NAMESPACE_PACKAGES list is empty")
        return 0

    # Target directory for mirrored package docs (symlinks or copies)
    target_base_dir = docs_dir / "contained_package_docs_mirror"
    target_base_dir.mkdir(exist_ok=True)

    # Copy or symlink referenced directories for each package
    cprint("Copying referenced directories from namespace packages...")
    total_copied = 0
    processed_packages = 0

    for namespace_package in namespace_packages:
        package_name = namespace_package.split('.')[-1]
        package_dir = project_root / "packages" / package_name

        if not package_dir.exists():
            cprint(f"Warning: Package directory not found: {package_dir}")
            continue

        copied_count = copy_package_referenced_dirs(
            package_name, package_dir, target_base_dir, cprint, mode=args.mode
        )
        total_copied += copied_count
        processed_packages += 1

    cprint(f"Copy operation complete!")
    cprint(f"Processed {processed_packages} packages, copied {total_copied} directories total")

    return 0


if __name__ == "__main__":
    sys.exit(main())
