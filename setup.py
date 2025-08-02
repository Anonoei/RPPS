from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

import numpy

Options.docstrings = True
Options.annotate = False

extensions = [
    Extension("*", ["src/**/*.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="rpps",
    ext_modules=cythonize(extensions, build_dir="build",
        compiler_directives={"language_level": 3, "profile": False}
    ),
)
