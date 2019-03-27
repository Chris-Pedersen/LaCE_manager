#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "User interface for the marginalized likelihood."

setup(name="user_interface", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/user_interface",
    author="Andreu Font-Ribera, Chris Pedersen, Keir Rogers",
    py_modules=['camb_cosmo','fit_pk_kms','lya_results'],
    package_dir={'': 'py'})

