from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tsdf-torch',
    version='0.01',
    description="TSDF-Fusion cuda extension for PyTorch",
    ext_modules=[
        CUDAExtension('tsdf_torch', [
            'fusion.cpp',
            'fusion_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)