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
        # PROTOTYPE (throwaway): Rust GT-only VCF parser, see rust/prototype_vcf_gt
        rust_extensions=[RustExtension('allel2._prototype_rust_vcf',
                                       path='rust/prototype_vcf_gt/Cargo.toml',
                                       binding=Binding.PyO3,
                                       # editable installs default to debug builds
                                       debug=False)],
        package_data={"allel2.test": ["data/*"]},
        zip_safe=False,
    )
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
