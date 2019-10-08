from distutils.core import setup
from Cython.Build import cythonize

setup(name='csvwriter_app',
      ext_modules=cythonize("csvwritercy.py"))