#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


requirements = [
    "torchvision",
    "ninja",
    "yacs",
    "cython",
    "matplotlib",
    "tqdm",
    "opencv-python",
    "scikit-image"
]


def get_extensions():
    extensions_dir = os.path.join("atss_core", "csrc")

    import shutil
    import torch.utils.cpp_extension as ext
    ninja_original = ext._write_ninja_file_and_compile_objects
    print("X!", ninja_original)

    def ninja_patch(sources, objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, build_directory, verbose, with_cuda):
        #headers = glob.glob(os.path.join(extensions_dir, "*.h"))
        headers = glob.glob(os.path.join(extensions_dir, "cpu/*.h"))
        for header in headers:
            print("x", header, os.path.join(build_directory, header))
            shutil.copyfile(header, os.path.join(build_directory, header))
        
        print("X2", ninja_original)
        return ninja_original(sources, objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, build_directory, verbose, with_cuda)

    ext._write_ninja_file_and_compile_objects = ninja_patch


    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "atss_core._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules

setup(
    name="atss",
    version="0.1",
    author="Shifeng Zhang",
    url="https://github.com/sfzhang15/ATSS",
    description="object detector in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    include_package_data=True,
)
