"""Generates a Latin hypercube, and generates corner plot."""

import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import configargparse
# our modules below
import camb_cosmo
import fit_linP
import sim_params_cosmo
import sim_params_space
import read_genic
import write_config
import latin_hypercube

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--basedir', type=str, help='Base directory where all sims will be stored (crashes if it already exists)',required=True)
parser.add_argument('--add_growth', action='store_true', help='Add parameter describing linear growth of structure',required=False)
parser.add_argument('--add_amplitude', action='store_true', help='Add parameter describing amplitude of linear power',required=False)
parser.add_argument('--add_slope', action='store_true', help='Add parameter describing slope of linear power',required=False)
parser.add_argument('--add_running', action='store_true', help='Add parameter describing running of linear power',required=False)
parser.add_argument('--add_heat_amp', action='store_true', help='Add parameter to boost heating',required=False)
parser.add_argument('--add_heat_slo', action='store_true', help='Add parameter to boost heating depending on density',required=False)
parser.add_argument('--add_z_rei', action='store_true', help='Add parameter to set hydrogen reionization (middle point)',required=False)
parser.add_argument('--nsamples', type=int, default=10, help='Number of samples in Latin hypercube')
parser.add_argument('--ngrid', type=int, default=64, help='Number of particles per side in simulation')
parser.add_argument('--box_Mpc', type=float, default=50.0, help='Simulation box size (in Mpc)')
parser.add_argument('--zs', type=str, help='Comma-separated list of redshifts (including last snapshot)')
parser.add_argument('--seed', type=int, default=123, help='Random seed to setup Latin hypercube')
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values()) 
print("----------")

verbose=args.verbose

# transform input string to list of sorted redshifts
if args.zs:
    zs=np.sort([float(z) for z in args.zs.split(',')])[::-1]
    print('will use input redshifts',zs)
else:
    zs=None

# setup parameter space
param_space=sim_params_space.SimulationParameterSpace(filename=args.config,
                    add_growth=args.add_growth,add_amplitude=args.add_amplitude,
                    add_slope=args.add_slope,add_running=args.add_running,
                    add_heat_amp=args.add_heat_amp,
                    add_heat_slo=args.add_heat_slo,
                    add_z_rei=args.add_z_rei)

# print parameter information
if verbose:
    print('z_star =',param_space.z_star)
    print('kp_Mpc =',param_space.kp_Mpc)
    for key,param in param_space.params.items():
        print(key,param)

# get parameter ranges
Npar=len(param_space.params)
param_limits=np.empty([Npar,2])
for key,param in param_space.params.items():
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

# make sure the base directory does not exist
basedir=args.basedir
if os.path.exists(basedir):
    raise ValueError(basedir+' already exists')
os.mkdir(basedir)

# write file with description of the hypercube
write_config.write_cube_json_file(basedir,param_space.params,cube)

for sample in range(nsamples):
    sim_params=cube[sample]
    if verbose: print(sample,sim_params)
    # setup cosmology from a given set of simulation parameters
    cosmo_sim=sim_params_cosmo.cosmo_from_sim_params(param_space,
            sim_params,verbose=verbose)
    if verbose: camb_cosmo.print_info(cosmo_sim)
    # figure out (medium) redshift of (hydrogen) reionization
    if 'z_rei' in param_space.params:
        ip=param_space.params['z_rei']['ip']
        z_rei=sim_params[ip]
    else:
        z_rei=9.0

    # figure out heating boost
    if 'heat_amp' in param_space.params:
        ip=param_space.params['heat_amp']['ip']
        heat_amp=sim_params[ip]
    else:
        heat_amp=1.0
    if 'heat_slo' in param_space.params:
        ip=param_space.params['heat_slo']['ip']
        heat_slo=sim_params[ip]
    else:
        heat_slo=0.0

    sim_dir=basedir+'/sim_pair_'+str(sample)+'/'
    os.mkdir(sim_dir)
    # make a different folder for each simulation in the pair
    plus_dir=sim_dir+'/sim_plus/'
    os.mkdir(plus_dir)
    minus_dir=sim_dir+'/sim_minus/'
    os.mkdir(minus_dir)


    # write treecool files for both simulations in pair
    write_config.write_treecool_file(plus_dir,z_mid_HI_reion=z_rei)
    write_config.write_treecool_file(minus_dir,z_mid_HI_reion=z_rei)

    # write GenIC and MP-Gadget parameters, for both simulations in pair
    if verbose: print('write config files for GenIC and Gadget')
    write_config.write_genic_file(plus_dir,cosmo_sim,
            Ngrid=args.ngrid,box_Mpc=args.box_Mpc,paired=False)
    zs=write_config.write_gadget_file(plus_dir,cosmo_sim,
            heat_amp=heat_amp,heat_slo=heat_slo,
            Ngrid=args.ngrid,zs=zs)
    write_config.write_genic_file(minus_dir,cosmo_sim,
            Ngrid=args.ngrid,box_Mpc=args.box_Mpc,paired=True)
    _=write_config.write_gadget_file(minus_dir,cosmo_sim,
            heat_amp=heat_amp,heat_slo=heat_slo,
            Ngrid=args.ngrid,zs=zs)

    # compute linear power in each snapshot and store in JSON format
    if verbose: print('write JSON file for simulation pair')
    write_config.write_sim_json_file(sim_dir,param_space,cosmo_sim,zs=zs)

print('finished')
