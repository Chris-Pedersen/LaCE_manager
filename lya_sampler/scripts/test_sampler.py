import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import configargparse
import corner
import cProfile
# our own modules
import simplest_emulator
import linear_emulator
import gp_emulator
import data_PD2013
import mean_flux_model
import thermal_model
import pressure_model
import lya_theory
import likelihood
import emcee_sampler


# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True,
        help='config file path')
parser.add_argument('--free_parameters', type=str, default='ln_tau_0, ln_tau_1',
        help='Comma-separated string of free parameters to use', required=False)
parser.add_argument('--nwalkers', type=int, default=None,
        help='Number of walkers in emcee (even integer)', required=False)
parser.add_argument('--nsteps', type=int, default=500,
        help='Number of steps in main chain', required=False)
parser.add_argument('--nburnin', type=int, default=100,
        help='Number of steps in burn-in phase', required=False)
parser.add_argument('--max_arxiv_size', type=int, default=None,
        help='Maximum number of models to train emulators', required=False)
parser.add_argument('--undersample_z', type=int, default=1, required=False,
        help='Undersample simulation snapshots, to make lighter emulators')
parser.add_argument('--emu_type', type=str, default='polyGP',
        help='Type of emulator to use (polyGP, kGP or linear)', required=False)
parser.add_argument('--chain_filename', type=str, default=None,
        help='Write the chain to this file (no extension)', required=False)
parser.add_argument('--verbose', action='store_true',
        help='Print runtime information',required=False)

args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values())
print("----------")

verbose=args.verbose

# read P1D measurement
data=data_PD2013.P1D_PD2013(blind_data=True)
zs=data.z

# read emulator
basedir='../../p1d_emulator/sim_suites/emulator_512_17052019/'
p1d_label='p1d'
skewers_label='Ns100_wM0.05'
if args.emu_type=='polyGP':
    if verbose: print('use polyGP emulator')
    # do not emulate growth or running (for now)
    #paramList=["mF","Delta2_p","n_p","sigT_Mpc","gamma","kF_Mpc"]
    paramList=None
    emu=gp_emulator.PolyfitGPEmulator(basedir,p1d_label,skewers_label,
                    max_arxiv_size=args.max_arxiv_size,
                    undersample_z=args.undersample_z,
                    paramList=paramList,kmax_Mpc=5,train=True)
elif args.emu_type=='kGP':
    if verbose: print('use kGP emulator')
    # do not emulate growth or running (for now)
    #paramList=["mF","Delta2_p","n_p","sigT_Mpc","gamma","kF_Mpc"]
    paramList=None
    emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                    max_arxiv_size=args.max_arxiv_size,
                    undersample_z=args.undersample_z,
                    paramList=paramList,kmax_Mpc=5,train=True)
elif args.emu_type=='linear':
    if verbose: print('use linear emulator')
    emu=linear_emulator.LinearEmulator(basedir,p1d_label,skewers_label,
                    max_arxiv_size=args.max_arxiv_size,
                    undersample_z=args.undersample_z)
else:
    raise ValueError('wrong emulator type '+args.emu_type)

# specify free parameters in likelihood (make sure there are no empty spaces)
free_parameters=[par.strip() for par in args.free_parameters.split(',')]
if verbose: print('input free parameters',free_parameters)

sampler = emcee_sampler.EmceeSampler(emulator=emu,nwalkers=args.nwalkers,
                        free_parameters=free_parameters,verbose=verbose)

for p in sampler.like.free_params:
    print(p.name,p.value)

# run burn-in
sampler.like.go_silent()
sampler.run_burn_in(nsteps=args.nburnin)
#cProfile.run("sampler.run_burn_in(nsteps=args.nburnin)",sort='cumtime')

# run main chain
sampler.run_chains(nsteps=args.nsteps)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(
                                    sampler.sampler.acceptance_fraction)))

# plot results
sampler.plot_corner(cube=True)
sampler.plot_corner(cube=False)

# write chain to file
if args.chain_filename:
    sampler.write_chain_to_file(args.chain_filename)
