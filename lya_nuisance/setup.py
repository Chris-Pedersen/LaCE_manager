#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Nuisance models in Lyman-alpha forest P1D analyses."

setup(name="lya_nuisance", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/lya_nuisance",
    author="Andreu Font-Ribera, Chris Pedersen, Keir Rogers",
    py_modules=['mean_flux_model','thermal_model'],
    package_dir={'': 'py'})

