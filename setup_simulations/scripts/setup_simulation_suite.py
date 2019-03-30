"""Generates a Latin hypercube, and generates corner plot."""

import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
# our modules below
import camb_cosmo
import fit_linP
import sim_params_cosmo
import sim_params_space
import read_genic
import write_config
import latin_hypercube

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, help='Base directory where all sims will be stored (crashes if it already exists)',required=True)
parser.add_argument('--configfile', type=str, help='Configuration file for this simulation suite',required=False)
parser.add_argument('--add_running', action='store_true', help='Add parameter describing running of linear power',required=False)
parser.add_argument('--add_mu_H', action='store_true', help='Add parameter to boost heating in Hydrogen',required=False)
parser.add_argument('--nsamples', type=int, default=10, help='Number of samples in Latin hypercube')
parser.add_argument('--seed', type=int, default=123, help='Random seed to setup Latin hypercube')
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

verbose=args.verbose

# setup parameter space
param_space=sim_params_space.SimulationParameterSpace(file_name=args.configfile,
                    add_running=args.add_running,add_mu_H=args.add_mu_H)
params=param_space.params

# get pivot point
z_star=params['Om_star']['z_star']
kp_Mpc=params['n_star']['kp_Mpc']

# print parameter information
if verbose:
    print('z_star =',z_star)
    print('kp_Mpc =',kp_Mpc)
    for key,param in params.items():
        print(key,param)

# get parameter ranges
Npar=len(params)
param_limits=np.empty([Npar,2])
for key,param in params.items():
    ip=param['ip']
    param_limits[ip][0]=param['min_val']
    param_limits[ip][1]=param['max_val']

# generate Latin hypercube 
nsamples=args.nsamples
seed=args.seed
cube=latin_hypercube.get_hypercube_samples(param_limits, nsamples, 
        prior_points = None, seed=seed)

# print information about cube
if verbose:
    print('# samples =',nsamples)
    print('random seed',seed)
    print('initial points in cube')
    print(cube)

# get fiducial cosmology
cosmo_fid = camb_cosmo.get_cosmology()
if verbose:
    camb_cosmo.print_info(cosmo_fid)

# setup fiducial linear power model
linP_model_fid=fit_linP.LinearPowerModel(cosmo_fid,z_star=z_star,
            k_units='Mpc',kp=kp_Mpc)
if verbose:
    print('fiducial linear power parameters',linP_model_fid.get_params())

# make sure the base directory does not exist
basedir=args.basedir
if os.path.exists(basedir):
    raise ValueError(basedir+' already exists')
os.mkdir(basedir)

# write file with description of the hypercube
write_config.write_cube_json_file(basedir,params)
for sample in range(nsamples):
    sim_params=cube[sample]
    if verbose:
        print(sample,sim_params)
    cosmo_sim=sim_params_cosmo.cosmo_from_sim_params(params,sim_params,
            linP_model_fid,verbose=verbose)

    sim_dir=basedir+'/sim_pair_'+str(sample)+'/'
    os.mkdir(sim_dir)
    # make a different folder for each simulation in the pair
    plus_dir=sim_dir+'/sim_plus/'
    os.mkdir(plus_dir)
    minus_dir=sim_dir+'/sim_minus/'
    os.mkdir(minus_dir)

    # write GenIC and MP-Gadget parameters, for both simulations in pair
    write_config.write_genic_file(plus_dir,cosmo_sim,paired=False)
    write_config.write_gadget_file(plus_dir,cosmo_sim)
    write_config.write_genic_file(minus_dir,cosmo_sim,paired=True)
    write_config.write_gadget_file(minus_dir,cosmo_sim)

    # construct linear power model and store in JSON format
    linP_model_sim=fit_linP.LinearPowerModel(cosmo_sim,z_star=z_star,
            k_units='Mpc',kp=kp_Mpc)

    write_config.write_sim_json_file(sim_dir,params,sim_params,linP_model_sim)

print('finished')
