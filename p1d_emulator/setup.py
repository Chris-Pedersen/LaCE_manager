#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Interpolate flux power measured in suite of simulations"

setup(name="p1d_emulator", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/p1d_emulator",
    author="Andreu Font-Ribera",
    py_modules=['p1d_arxiv','poly_p1d','simplest_emulator','linear_emulator',
            'gp_emulator','mf_emulator','z_emulator','test_simulation'],
    package_dir={'': 'py'})

