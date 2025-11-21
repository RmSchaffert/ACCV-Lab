from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
import sys
from pathlib import Path

from accvlab_build_config import run_external_build


def get_extensions():
    """Return all extensions"""
    extensions = []
    return extensions


# Run external build before setup
run_external_build(Path(__file__).parent)

setup(
    name="accvlab.optim_test_tools",
    version="0.1.0",
    description="Optimization Testing Tools Package (part of the ACCV-Lab package).",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    zip_safe=False,
)
