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
Script to generate documentation structure for new ACCV-Lab namespace packages.
This script only creates files that don't already exist - it's safe to run repeatedly.
"""

import sys
import argparse
import functools
from pathlib import Path
from textwrap import dedent


def vprint(*args, verbose=True, **kwargs):
    """Print only if verbose mode is enabled"""
    if verbose:
        print(*args, **kwargs)


def create_namespace_package_folder_structure(subpackage_name, package_docs_dir, cprint):
    """Create a folder structure with basic files for a subpackage"""

    def create_intro_template(package_name, package_dir):
        """Create a very basic intro template that should be filled out"""

        if (package_dir / "intro.rst").exists():
            return

        content = dedent(f"""
            Introduction
            ============
            
            This is the documentation for the **{package_name}** package.
            
            .. note::
               This is a placeholder page. Please fill out this documentation e.g. with:
               
               * Package overview and purpose
               * Basic usage examples
               * Key features and capabilities
            """).strip()

        with open(package_dir / "intro.rst", 'w') as f:
            f.write(content)

    def create_api_reference(package_name, full_name, package_dir):
        """Create API reference documentation"""
        api_file = package_dir / "api.rst"
        if api_file.exists():
            return

        content = dedent(f"""
            API Reference
            =============
            
            Complete API documentation for the {package_name} package.
            
            .. currentmodule:: {full_name}
            
            .. automodule:: {full_name}
               :members:
               :undoc-members:
               :show-inheritance:
               :special-members: __init__
            """).strip()

        with open(api_file, 'w') as f:
            f.write(content)

    def create_package_index(package_name, package_dir):
        """Create a simple index file for a subpackage"""
        index_file = package_dir / "index.rst"
        if index_file.exists():
            return

        title = f"{package_name.title()} Package"

        content = dedent(f"""
            {title}
            {'=' * len(title)}
            
            Documentation for the {package_name} subpackage.
            
            .. toctree::
               :maxdepth: 1
            
               intro
               api
            """).strip()

        with open(index_file, 'w') as f:
            f.write(content)

    package_name = subpackage_name.split('.')[-1]

    # Create docs directory if it doesn't exist
    package_docs_dir.mkdir(exist_ok=True)

    if not (package_docs_dir / "index.rst").exists():
        # Create index.rst (main entry point)
        create_package_index(package_name, package_docs_dir)

        # Create API reference (auto-generated)
        create_api_reference(package_name, subpackage_name, package_docs_dir)

        # Create a simple intro template if it doesn't exist
        create_intro_template(package_name, package_docs_dir)
    else:
        cprint(f"  Documentation index (`index.rst`) already exists for: {package_docs_dir}")
        cprint(f"  Skipping creation of `index.rst`, `api.rst`, and `intro.rst` files")

    return package_docs_dir


def main():
    """Main function to generate documentation structure for new namespace packages"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate documentation structure for new ACCV-Lab namespace packages'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
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
        print(
            "Please ensure namespace_packages_config.py exists in the project root with a NAMESPACE_PACKAGES list, for example:"
        )
        print("NAMESPACE_PACKAGES = ['accvlab.examples', 'accvlab.other_package']")
        return 1
    except AttributeError:
        print("Error: NAMESPACE_PACKAGES not found in namespace_packages_config.py")
        print("Please add a NAMESPACE_PACKAGES list to namespace_packages_config.py, for example:")
        print("NAMESPACE_PACKAGES = ['accvlab.examples', 'accvlab.other_package']")
        return 1

    if not namespace_packages:
        print("Error: NAMESPACE_PACKAGES list is empty in namespace_packages_config.py")
        print("Please add namespace packages to the NAMESPACE_PACKAGES list in namespace_packages_config.py")
        return 1

    # Generate namespace package documentation files in each package's docs folder
    cprint("Generating missing namespace package documentation...")
    created_count = 0
    for namespace_package in namespace_packages:
        package_name = namespace_package.split('.')[-1]

        # Find the package directory in the packages folder
        package_dir = project_root / "packages" / package_name
        if not package_dir.exists():
            cprint(f"  Warning: Package directory not found: {package_dir}")
            continue

        # Target the docs folder within each package
        package_docs_dir = package_dir / "docs"

        if not package_docs_dir.exists():
            create_namespace_package_folder_structure(namespace_package, package_docs_dir, cprint)
            cprint(f"  Created new: {package_docs_dir}")
            created_count += 1
        else:
            # Still run the function to create any missing individual files
            old_count = len(list(package_docs_dir.glob("*.rst")))
            create_namespace_package_folder_structure(namespace_package, package_docs_dir, cprint)
            new_count = len(list(package_docs_dir.glob("*.rst")))
            if new_count > old_count:
                cprint(f"  Added missing files to: {package_docs_dir}")
                created_count += 1
            else:
                cprint(f"  Already exists: {package_docs_dir}")

    if created_count == 0:
        cprint("All namespace package documentation already exists!")
    else:
        print(f"Documentation structure generation complete! Created/updated {created_count} packages.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
