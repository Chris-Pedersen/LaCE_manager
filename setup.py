#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Lyman-alpha forest flux power spectrum emulator"
version="1.0.1"

setup(name="lace_manager",
    version=version,
    description=description,
    url="https://github.com/Chris-Pedersen/LaCE_manager",
    author="Chris Pedersen, Andreu Font-Ribera et al.",
    author_email="c.pedersen@nyu.edu",
    packages=find_packages(),
    )
