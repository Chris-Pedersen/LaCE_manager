#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Lyman-alpha forest flux power spectrum emulator"
version="0.1.0"

setup(name="LyaCosmoParams",
    version=version,
    description=description,
    url="https://github.com/andreufont/LyaCosmoParams",
    author="Chris Pedersen, Andreu Font-Ribera et al.",
    author_email="christian.pedersen.17@ucl.ac.uk",
    packages=find_packages(),
    )
