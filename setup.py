# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import os
import numpy # Import numpy
from setuptools_rust import Binding, RustExtension

# find all the cython files and define extensions
extensions = []
for root, _, files in os.walk('allel2'):
    for filename in files:
        if filename.endswith('.pyx'):
            filepath = os.path.join(root, filename)
            modname = filepath.replace(os.path.sep, '.').strip('.pyx')
            extensions.append(
                Extension(
                    modname,
                    [filepath],
                    include_dirs=[numpy.get_include()] # Add numpy include path
                )
            )

def setup_package():
    metadata = dict(
        # note: all metadata is now in pyproject.toml
        # we need to retain this file for the cython extensions
        ext_modules=extensions,
        rust_extensions=[
            RustExtension('allel2.opt.io_vcf_read_rs',
                          path='rust/io_vcf_read/Cargo.toml',
                          binding=Binding.PyO3,
                          # editable installs default to debug builds, several
                          # times slower than the Cython parser
                          debug=False),
        ],
        package_data={"allel2.test": ["data/*"]},
        zip_safe=False,
    )
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
