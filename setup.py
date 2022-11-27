import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CUDAExtension)
from torch_spsr import __version__


if CUDA_HOME is None:
    print("Please install nvcc for your PyTorch distribution and set CUDA_HOME environment variable.")
    sys.exit(-1)


if sys.platform != "linux":
    print("This repository only supports x86-64 Linux!")


def get_extensions():
    extensions = [
        CUDAExtension(
            f'torch_spsr._common',
            ['csrc/common/bind.cpp', 'csrc/common/hashmap_cuda.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            f'torch_spsr._integration',
            ['csrc/integration/bind.cpp', 'csrc/integration/ddi.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            f'torch_spsr._marching_cubes',
            ['csrc/marching_cubes/bind.cpp',
             'csrc/marching_cubes/primal_mc.cu',
             'csrc/marching_cubes/dual_mc.cu',
             'csrc/marching_cubes/dual_mc_inds.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            f'torch_spsr._sparse_op',
            ['csrc/sparse_op/bind.cpp', 'csrc/sparse_op/convolution_cuda.cu',
             'csrc/sparse_op/plane_fitting.cu', 'csrc/sparse_op/sparse_product.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            f'torch_spsr._sparse_solve',
            ['csrc/sparse_solve/bind.cpp', 'csrc/sparse_solve/err_check.cu',
             'csrc/sparse_solve/solve_cusparse.cu', 'csrc/sparse_solve/solve_pcg.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']},
            libraries=['cusolver', 'cusparse']
        )
    ]

    return extensions


setup(
    name='torch_spsr',
    version=__version__,
    description='PyTorch Extension Library of Screened Poisson Surface Reconstruction',
    author='Jiahui Huang',
    author_email='huangjh.work@outlook.com',
    keywords=['pytorch', 'spsr', '3d', 'reconstruction'],
    python_requires='>=3.7',
    install_requires=[],
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
    include_package_data=True,
)
