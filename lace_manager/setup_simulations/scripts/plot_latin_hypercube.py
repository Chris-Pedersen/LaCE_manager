"""Generates a Latin hypercube, and generates corner plot."""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
# our modules below
import sim_params_space
import latin_hypercube
import corner

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--paramfile', type=str, help='file with parameter space',required=False)
parser.add_argument('--add_slope', action='store_true', help='add parameter describing slope of linear power',required=False)
parser.add_argument('--add_running', action='store_true', help='add parameter describing running of linear power',required=False)
parser.add_argument('--add_mu_H', action='store_true', help='add parameter to boost heating in Hydrogen',required=False)
parser.add_argument('--nsamples', type=int, default=10, help='Number of samples in Latin hypercube')
parser.add_argument('--seed', type=int, default=123, help='Random seed to setup Latin hypercube')
parser.add_argument('--plotfile', type=str, default='latin_hypercube', help='Name of plot file (without extension)')
parser.add_argument('--verbose', action='store_true', help='print class runtime information',required=False)
args = parser.parse_args()

verbose=args.verbose

# setup parameter space
param_space=sim_params_space.SimulationParameterSpace(file_name=args.paramfile,
                    add_slope=args.add_slope,add_running=args.add_running,
                    add_mu_H=args.add_mu_H)
params=param_space.params

# get pivot point
z_star=params['Om_star']['z_star']
kp_Mpc=params['Delta2_star']['kp_Mpc']

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

labels=['']*Npar
for key,param in params.items():
    ip=param['ip']
    labels[ip]=param['latex']    

fig=corner.corner(cube,labels=labels)
plt.savefig(args.plotfile+'.pdf')
plt.show()
