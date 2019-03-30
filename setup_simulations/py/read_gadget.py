"""Read MP-Gadget configuration file. File addapted from code by Simeon Bird."""

import numpy as np
import argparse
import configobj
import validate

# define variables and default values in Gadget
Gadget_configspec = """
OutputList = string(default='')
Nmesh = integer(default=1)
    Omega0 = float(0,1)
    OmegaLambda = float(0,1)
    OmegaBaryon = float(0,1,default=0.0486)
    HubbleParam = float(0,2)
    MNue = float(min=0, default=0)
    MNum = float(min=0, default=0)
    MNut = float(min=0, default=0)""".split('\n') 

def _check_gadget_config(config):
    """Check that the MP-Gadget config file is sensible."""
    vtor = validate.Validator()
    config.validate(vtor)
    # Check unsupported configurations
    if config['MNue']+config['MNum']+config['MNut'] > 0.:
        raise ValueError("Simulations should not have neutrions")

def read_gadget_paramfile(paramfile, verbose=False):
    """Parse a MP-Gadget parameter file and returns a dictionary"""

    config = configobj.ConfigObj(infile=paramfile, configspec=Gadget_configspec,
                    file_error=True)
    # check file is healthy
    _check_gadget_config(config)
    if verbose:
        print('successfully read healthy configuration file')
    return config

def snapshot_redshifts(config):
    scale_factors=[float(astr) for astr in config['OutputList'].split(',')]
    # add last snapshot, when simulation ends
    scale_factors.append(float(config['TimeMax']))
    zs=1.0/np.array(scale_factors)-1.0
    return zs

def redshifts_from_paramfile(paramfile, verbose=False):
    config=read_gadget_paramfile(paramfile, verbose)
    return snapshot_redshifts(config)

