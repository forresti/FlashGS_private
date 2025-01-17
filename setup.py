import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)


def _is_cuda() -> bool:
    return torch.version.cuda is not None


ext_modules = []

# Compiler flags.
CXX_FLAGS = ["-g", "-O1", "-std=c++11"]
NVCC_FLAGS = ["-O1", "-lineinfo"]


def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

if _is_cuda():
    ext_modules.append(
        CUDAExtension(
            name="flash_gaussian_splatting",
            sources=glob("csrc/cuda_rasterizer/*.cu") + glob("csrc/pybind.cpp"),
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
    )

setup(name='flash_gaussian_splatting',
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension}
      )
