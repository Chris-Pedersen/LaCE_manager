import numpy as np
import sys
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import emcee
import corner
import configargparse
import shutil
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
import data_MPGADGET
import z_emulator
import p1d_arxiv

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--basedir', help='Which emulator to load')
parser.add_argument('--skewers_label', help='Add parameter describing linear growth of structure')
parser.add_argument('--test_sim_number',type=int, help='Which sim number to use as mock data')
parser.add_argument('--kmax_Mpc',type=float, help='Maximum k to train emulator')
parser.add_argument('--undersample_z',type=int, help='Undersample redshifts')
parser.add_argument('--z_max',type=float, help='Maximum redshift')
parser.add_argument('--drop_tau_rescalings',action='store_true', help='Drop mean flux rescalings')
parser.add_argument('--drop_temp_rescalings',action='store_true', help='Drop temperature rescalings')
parser.add_argument('--nearest_tau', action='store_true',help='Keep only nearest tau rescaling? Only used when tau rescalings are included')
parser.add_argument('--undersample_cube',type=int, help='Undersample the Latin hypercube of training sims')
parser.add_argument('--z_emulator',action='store_true',help='Whether or not to use a single GP on each redshfit bin')
parser.add_argument('--free_parameters', nargs="+", help='List of parameters to sample')
parser.add_argument('--nwalkers', type=int,help='Number of walkers to sample')
parser.add_argument('--nsteps', type=int,help='Max number of steps to run (assuming we dont reach autocorrelation time convergence')
parser.add_argument('--burn_in', type=int,help='Number of burn in steps')
parser.add_argument('--prior_Gauss_rms',type=float, help='Width of Gaussian prior')
parser.add_argument('--emu_cov_factor', type=float,help='Factor between 0 and 1 to vary the contribution from emulator covariance')
parser.add_argument('--emu_noise_var', type=float,help='Emulator noise variable')
args = parser.parse_args()

test_sim_number=args.test_sim_number

# read P1D measurement
#z_list=np.array([2.0,2.75,3.25,4.0])
data=data_MPGADGET.P1D_MPGADGET(sim_number=test_sim_number)
zs=data.z

repo=os.environ['LYA_EMU_REPO']
skewers_label=args.skewers_label
#skewers_label=None
basedir=repo+args.basedir
#basedir=repo+"/p1d_emulator/sim_suites/emulator_256_15072019/"
p1d_label=None
undersample_z=args.undersample_z
paramList=args.free_parameters
free_parameters=args.free_parameters
kmax_Mpc=args.kmax_Mpc

archive=p1d_arxiv.ArxivP1D(basedir=basedir,
                            drop_tau_rescalings=True,z_max=args.z_max,
                            drop_sim_number=test_sim_number,nearest_tau=True,
                            drop_temp_rescalings=True,skewers_label=skewers_label,
                            undersample_cube=1)


if args.z_emulator==False:
    emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,z_max=args.z_max,
                                    passArxiv=archive,
                                    verbose=False,paramList=paramList,train=True,
                                    emu_type="k_bin", checkHulls=False,set_noise_var=args.emu_noise_var,
                                    drop_tau_rescalings=args.drop_tau_rescalings,
                                    drop_temp_rescalings=args.drop_temp_rescalings)
else:
    emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,z_max=args.z_max,
                                    verbose=False,paramList=paramList,train=True,
                                    emu_type="k_bin",passArxiv=archive,checkHulls=False,
                                    set_noise_var=args.emu_noise_var,
                                    drop_tau_rescalings=args.drop_tau_rescalings,
                                    drop_temp_rescalings=args.drop_temp_rescalings)



like=likelihood.simpleLikelihood(data=data,emulator=emu,
                            free_parameters=free_parameters,verbose=False,
                            prior_Gauss_rms=args.prior_Gauss_rms,
                            emu_cov_factor=args.emu_cov_factor)

#like.plot_p1d()

sampler = emcee_sampler.EmceeSampler(like=like,
                        free_parameters=free_parameters,verbose=True,
                        nwalkers=args.nwalkers,progress=False)


for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)


sampler.like.go_silent()
sampler.store_distances=True
sampler.run_burn_in(nsteps=args.burn_in)

sampler.run_chains(nsteps=args.nsteps)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.sampler.acceptance_fraction)))


sampler.write_chain_to_file()
sampler.plot_corner(cube=False,mock_values=True)
sampler.plot_best_fit()

## Finally copy the config file to the save folder
shutil.copy(sys.argv[2],sampler.save_directory+"/"+sys.argv[2])
