#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "Setup of hydro simulations for Lyman-alpha cosmology."

setup(name="setup_sims", 
    version="0.1.0",
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams/tree/master/setup_simulations",
    author="Andreu Font-Ribera",
    py_modules=['read_genic','read_gadget','write_config',
                'latin_hypercube','sim_params_cosmo',
                'sim_params_space','write_submit_simulation_dirac'],
    package_dir={'': 'py'})

