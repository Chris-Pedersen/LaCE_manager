#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Cosmology tools used in Lyman-alpha forest analyses."

setup(name="lya_cosmo", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/lya_cosmo",
    author="Andreu Font-Ribera",
    py_modules=['camb_cosmo','fit_linP','recons_cosmo'],
    package_dir={'': 'py'})

