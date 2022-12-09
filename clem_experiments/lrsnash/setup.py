from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# Specify all sources in Extension object
ext = Extension("lrsnash",sources=[
        "lrsnash.pyx", 
        "lib/lrsnash.c"
    ])

setup(
    ext_modules = cythonize(ext),
    include_dirs=[numpy.get_include()]
)