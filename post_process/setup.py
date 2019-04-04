#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Read Gadget snapshots and extract Lya skewers"

setup(name="post_process", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/post_process",
    author="Andreu Font-Ribera, Chris Pedersen",
    py_modules=['extract_skewers','temperature_density',
                'snapshot_admin','measure_flux_power',
                'write_submit_skewers_darwin'],
    package_dir={'': 'py'})

