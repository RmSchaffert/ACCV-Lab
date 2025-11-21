#!/usr/bin/env python3
"""
Simple Example for ACCV-Lab Example Package.

This example demonstrates basic usage of the example package functions.
It's designed to show how examples can be included in documentation.
"""

import accvlab.example_package as example_pkg

# @NOTE
# This note will be highlighted in the documentation using the note-literalinclude directive (see the
# Documentation Setup Guide for more details).


def simple_example():
    """Demonstrate simple function usage."""
    print("=== Simple Example ===")

    # Use the hello function
    message = example_pkg.hello_examples()
    print(f"Message: {message}")


if __name__ == "__main__":
    simple_example()
