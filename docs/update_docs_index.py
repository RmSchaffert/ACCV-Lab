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
Script to update the main documentation index with current namespace packages.
This script should be run every time documentation is generated to ensure
the main index.rst file reflects the current namespace packages configuration.
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


def extract_toctree_header_config(content):
    """Extract the toctree header configuration from the RST comment section"""
    # Find the RST comment section that contains the header configuration
    comment_start_marker = ".. The following defines the doctree header for the namespace packages section."
    comment_start = content.find(comment_start_marker)
    if comment_start == -1:
        return None

    # Find the end of the RST comment (next non-indented line that's not part of the comment)
    lines = content[comment_start:].split('\n')
    comment_lines = []
    comment_lines.append(lines[0])  # First line with the ".."

    # Collect all indented lines that are part of the comment
    for i in range(1, len(lines)):
        line = lines[i]
        # RST comment continues if line is indented or empty
        if line.startswith('   ') or line.strip() == '':
            comment_lines.append(line)
        else:
            # Found non-indented line, comment ends here
            break

    comment_content = '\n'.join(comment_lines)

    # Find the content between the ===== markers
    start_marker = "===== start ====="
    end_marker = "===== end ====="

    start_pos = comment_content.find(start_marker)
    if start_pos == -1:
        return None

    # Find the second =====
    end_pos = comment_content.find(end_marker, start_pos + len(start_marker))
    if end_pos == -1:
        return None

    # Extract the toctree header configuration
    header_config_raw = comment_content[start_pos + len(start_marker) : end_pos]

    # Clean up the header config by preserving leading spaces for indentation
    lines = header_config_raw.split('\n')
    cleaned_lines = []

    # Find first non-empty line to determine base indentation
    base_indent = 0
    for line in lines:
        if line.strip():
            base_indent = len(line) - len(line.lstrip())
            break

    # Process all lines removing base indentation
    for line in lines:
        if line.strip():
            # Remove base indentation and trailing whitespace
            cleaned_line = line[base_indent:].rstrip()
            cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)


def extract_template_line(content):
    """Extract the TEMPLATE line for package links from the RST comment section"""
    for line in content.splitlines():
        if line.strip().startswith('.. TEMPLATE:'):
            # Extract the template after the colon and strip whitespace
            return line.split(':', 1)[1].strip()
    return None


def generate_main_index(namespace_packages, docs_dir, cprint_func):
    """Generate the main index.rst by updating the auto-generated section between comments"""

    output_file = docs_dir / "index.rst"

    if not output_file.exists():
        print(f"Error: {output_file} not found, cannot update auto-generated section")
        return False

    # Read the current index.rst
    with open(output_file, 'r') as f:
        content = f.read()

    # Extract the toctree header configuration
    header_config = extract_toctree_header_config(content)
    if not header_config:
        print("Warning: Could not find toctree header configuration, using default")
        header_config = dedent("""
            .. toctree::
               :maxdepth: 2
               :caption: Namespace Packages:
            """).strip()

    # Extract the template line for package links
    template_line = extract_template_line(content)
    if not template_line:
        print("Error: Could not find TEMPLATE line for package links in index.rst.")
        print("Please add the following line to index.rst:")
        print("   .. TEMPLATE: contained_package_docs_mirror/{package}/docs/index")
        print("The line should be placed in the comment section before the auto-generated section.")
        return False

    # Generate the namespace packages toctree entries using the template
    namespace_packages_entries = []
    for package in namespace_packages:
        package_name = package.split('.')[-1]
        entry = template_line.replace('{package}', package_name)
        namespace_packages_entries.append(f"   {entry}")

    # Create the complete toctree section using the extracted header
    if namespace_packages_entries:
        toctree_content = header_config + '\n\n' + '\n'.join(namespace_packages_entries)
    else:
        toctree_content = header_config + '\n\n   .. note:: No namespace packages configured.'

    # Define the RST comment markers
    start_marker = ".. AUTO-GENERATED NAMESPACE PACKAGES START - DO NOT EDIT MANUALLY"
    end_marker = ".. AUTO-GENERATED NAMESPACE PACKAGES END - DO NOT EDIT MANUALLY"

    # Find the auto-generated section
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)

    if start_pos == -1 or end_pos == -1:
        print(f"Error: Could not find auto-generated section markers in {output_file}")
        print("Please ensure the file has the following RST comment markers:")
        print(f"   {start_marker}")
        print(f"   {end_marker}")
        return False

    # Replace the content between the markers
    before_section = content[: start_pos + len(start_marker)]
    after_section = content[end_pos:]

    # Construct the new content
    new_content = before_section + '\n\n' + toctree_content + '\n\n' + after_section

    # Write the updated index.rst
    with open(output_file, 'w') as f:
        f.write(new_content)

    cprint_func(
        f"Updated main index.rst with {len(namespace_packages)} namespace packages using extracted header configuration"
    )
    return True


def main():
    """Main function to update the documentation index"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Update the main documentation index with current namespace packages'
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
        cprint("Warning: NAMESPACE_PACKAGES list is empty in namespace_packages_config.py")
        cprint("The main index will show 'No namespace packages configured.'")
        namespace_packages = []  # Continue with empty list

    # Update main index.rst
    cprint("Updating main documentation index...")
    success = generate_main_index(namespace_packages, docs_dir, cprint)

    if success:
        print("Documentation index update complete!")
        return 0
    else:
        print("Failed to update documentation index!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
