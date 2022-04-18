#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Lyman-alpha forest flux power spectrum emulator"
version="1.0.1"

setup(name="lace_manager",
    version=version,
    description=description,
    url="https://github.com/Chris-Pedersen/LaCE",
    author="Chris Pedersen, Andreu Font-Ribera et al.",
    author_email="christian.pedersen.17@ucl.ac.uk",
    packages=find_packages(),
    )
