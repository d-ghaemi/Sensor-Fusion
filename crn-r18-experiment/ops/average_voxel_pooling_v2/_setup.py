
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name="average_voxel_pooling_ext",
    ext_modules=[CUDAExtension(name="average_voxel_pooling_ext", sources=["src/average_voxel_pooling_forward.cpp", "src/average_voxel_pooling_forward_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
