#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "MCMC sampler for Lyman alpha likelihood"

setup(name="lya_sampler", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/lya_sampler",
    author="Andreu Font-Ribera",
    py_modules=['emcee_sampler'],
    package_dir={'': 'py'})

