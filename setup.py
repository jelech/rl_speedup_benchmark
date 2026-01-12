from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "src.envs.cartpole.cy_impl",
        ["src/envs/cartpole/cy_impl.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "src.envs.boids.cy_impl",
        ["src/envs/boids/cy_impl.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
