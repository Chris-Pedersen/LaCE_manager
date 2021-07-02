import matplotlib ## Suppresses plotting issues on compute nodes
matplotlib.use("Agg")
import sys
import os
import configargparse
import shutil
import time
# our own modules
from lace.data import data_MPGADGET
from lace.emulator import gp_emulator
from lace.emulator import z_emulator
from lace.emulator import p1d_archive
from lace.likelihood import likelihood
from lace.sampler import emcee_sampler

""" Example script to run an emcee chain. The timeout flag at
sampler.run_sampler() will set a max time limit to save the chain
before a job hits the walltime limit """

os.environ["OMP_NUM_THREADS"] = "1"

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--basedir', help='Which emulator to load')
parser.add_argument('--emu_type', help='k_bin or polyfit emulator')
parser.add_argument('--skewers_label', help='Add parameter describing linear growth of structure')
parser.add_argument('--test_sim_number', help='Which sim number to use as mock data')
parser.add_argument('--kmax_Mpc',type=float, help='Maximum k to train emulator')
parser.add_argument('--undersample_z',type=int, help='Undersample redshifts')
parser.add_argument('--z_max',type=float, help='Maximum redshift')
parser.add_argument('--drop_tau_rescalings',action='store_true', help='Drop mean flux rescalings')
parser.add_argument('--drop_temp_rescalings',action='store_true', help='Drop temperature rescalings')
parser.add_argument('--nearest_tau', action='store_true',help='Keep only nearest tau rescaling? Only used when tau rescalings are included')
parser.add_argument('--asym_kernel', action='store_true',help='Use asymmetric, rbf-only kernel for the GP')
parser.add_argument('--undersample_cube',type=int, help='Undersample the Latin hypercube of training sims')
parser.add_argument('--z_emulator',action='store_true',help='Whether or not to use a single GP on each redshfit bin')
parser.add_argument('--free_parameters', nargs="+", help='List of parameters to sample')
parser.add_argument('--nwalkers', type=int,help='Number of walkers to sample')
parser.add_argument('--nsteps', type=int,help='Max number of steps to run (assuming we dont reach autocorrelation time convergence')
parser.add_argument('--burn_in', type=int,help='Number of burn in steps')
parser.add_argument('--prior_Gauss_rms',type=float, help='Width of Gaussian prior')
parser.add_argument('--emu_cov_factor', type=float,help='Factor between 0 and 1 to vary the contribution from emulator covariance')
parser.add_argument('--emu_noise_var', type=float,help='Emulator noise variable')
parser.add_argument('--parallel',action='store_true',help='Run sampler in parallel?')
parser.add_argument('--data_cov_factor',type=float,help='Factor to multiply the data covariance by')
parser.add_argument('--data_year', help='Which version of the data covmats and k bins to use, PD2013 or Chabanier2019')
parser.add_argument('--subfolder',default=None, help='Subdirectory to save chain file in')
parser.add_argument('--pivot_scalar',default=0.05,type=float, help='Primordial power spectrum pivot scale in 1/Mpc')
parser.add_argument('--include_CMB',action='store_true', help='Include CMB information?')
parser.add_argument('--reduced_IGM',action='store_true', help='Reduce IGM marginalisation in the case of use_compression=3?')
parser.add_argument('--use_compression',type=int, help='Go through compression parameters?')
args = parser.parse_args()

test_sim_number=args.test_sim_number
## Make sure test sim is dropped from training data
## in the case of running on one of the LH sims
if test_sim_number.isdigit():
    test_sim_number=int(test_sim_number)

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values()) 
print("----------")

## configargparse cannot accept nested lists or dictionaries
## so no elegant solution to passing a prior volume right now
## these are still saved with the sampler so no book-keeping issues though

## Example for sampling CMB parameters:
free_param_limits=[[0.0099,0.0109],
                [1.1e-09, 3.19e-09],
                [0.89, 1.05],
                [0.018, 0.026],
                [0.1,0.13],
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4],
                [-0.4, 0.4]]


''' ## Some template limits below
## for reference, the default primordial limits I have been using are
## (for a pivot_scalar of 0.7)
## [[1.1e-09, 3.19e-09], [0.89, 1.05],
## And for compressed params,
## [["Delta2_star", 0.24, 0.47], ["n_star", -2.352, -2.25]]
free_param_limits=[[1.1e-09, 3.19e-09], [0.89, 1.05],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4]]
'''

skewers_label=args.skewers_label
p1d_label=None
undersample_z=args.undersample_z
paramList=None
free_parameters=args.free_parameters
kmax_Mpc=args.kmax_Mpc

prior=args.prior_Gauss_rms
if prior==-1:
    prior=None

## Generate mock P1D measurement
data=data_MPGADGET.P1D_MPGADGET(sim_label=test_sim_number,
                                basedir=args.basedir,
                                skewers_label=args.skewers_label,
				                zmax=args.z_max,
                                data_cov_factor=args.data_cov_factor,
                                data_cov_label=args.data_year,
                                pivot_scalar=args.pivot_scalar)
zs=data.z

## Set up emulator training data
archive=p1d_archive.archiveP1D(basedir=args.basedir,
                            drop_tau_rescalings=args.drop_tau_rescalings,z_max=args.z_max,
                            drop_sim_number=test_sim_number,nearest_tau=args.nearest_tau,
                            drop_temp_rescalings=args.drop_temp_rescalings,skewers_label=skewers_label,
                            undersample_cube=args.undersample_cube,undersample_z=args.undersample_z)

## Set up an emulator
if args.z_emulator==False:
    emu=gp_emulator.GPEmulator(args.basedir,p1d_label,skewers_label,z_max=args.z_max,
                                    passarchive=archive,
                                    verbose=False,paramList=paramList,train=True,
                                    emu_type=args.emu_type, checkHulls=False,kmax_Mpc=kmax_Mpc,
                                    asymmetric_kernel=args.asym_kernel,rbf_only=args.asym_kernel,
                                    drop_tau_rescalings=args.drop_tau_rescalings,
                                    drop_temp_rescalings=args.drop_temp_rescalings,
				                    set_noise_var=args.emu_noise_var)
else:
    emu=z_emulator.ZEmulator(args.basedir,p1d_label,skewers_label,z_max=args.z_max,
                                    verbose=False,paramList=paramList,train=True,
                                    emu_type=args.emu_type,passarchive=archive,checkHulls=False,
                                    kmax_Mpc=kmax_Mpc,
                                    drop_tau_rescalings=args.drop_tau_rescalings,
                                    drop_temp_rescalings=args.drop_temp_rescalings,
				                    set_noise_var=args.emu_noise_var)

## Create likelihood object from data and emulator
like=likelihood.Likelihood(data=data,emulator=emu,
                            free_param_names=free_parameters,
			                free_param_limits=free_param_limits,
			                verbose=False,
                            prior_Gauss_rms=prior,
                            emu_cov_factor=args.emu_cov_factor,
                            pivot_scalar=args.pivot_scalar,
                            include_CMB=args.include_CMB,
                            use_compression=args.use_compression,
                            reduced_IGM=args.reduced_IGM)


## Pass likelihood to sampler
sampler = emcee_sampler.EmceeSampler(like=like,
                        free_param_names=free_parameters,verbose=False,
                        nwalkers=args.nwalkers,
                        subfolder=args.subfolder)

## Copy the config file to the save folder
shutil.copy(sys.argv[2],sampler.save_directory+"/"+sys.argv[2])

for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)

## Cannot call self.log_prob using multiprocess.pool
def log_prob(theta):
    return sampler.like.log_prob_and_blobs(theta)

start = time.time()
sampler.like.go_silent()
sampler.run_sampler(args.burn_in,args.nsteps,log_prob,parallel=args.parallel,timeout=47.)
end = time.time()
multi_time = end - start
print("Sampling took {0:.1f} seconds".format(multi_time))

sampler.write_chain_to_file()

## Copy corresponding job files to save folder
jobstring=jobstring="job"+os.environ['SLURM_JOBID']+".out"
slurmstring="slurm-"+os.environ['SLURM_JOBID']+".out"
shutil.copy(jobstring,sampler.save_directory+"/"+jobstring)
shutil.copy(slurmstring,sampler.save_directory+"/"+slurmstring)

