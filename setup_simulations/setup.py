#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Setup of hydro simulations for Lyman-alpha cosmology."

setup(name="setup_sims", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/setup_simulations",
    author="Andreu Font-Ribera, Chris Pedersen, Keir Rogers",
    py_modules=['read_genic','write_config',
                'latin_hypercube','sim_params_cosmo'],
    package_dir={'': 'py'})

