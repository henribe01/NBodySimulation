from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "nbody.barnes_hut_cython",
        ["nbody/barnes_hut_cython.pyx"],
        extra_compile_args=["-O3", "-ffast-math"], # Adjust for Windows if necessary (/O2)
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)