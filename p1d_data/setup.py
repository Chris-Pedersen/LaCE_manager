#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Read measured 1D power spectra and its covariances"

setup(name="p1d_data", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/p1d_data",
    author="Andreu Font-Ribera",
    py_modules=['base_p1d_data','data_PD2013','data_Chabanier2019','data_MPGADGET'],
    package_dir={'': 'py'})

