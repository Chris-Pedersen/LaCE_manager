import matplotlib.pyplot as plt
import numpy as np
import time
# our own modules
from LyaCosmoParams.emulator import gp_emulator
from LyaCosmoParams.likelihood import likelihood
from LyaCosmoParams.sampler import emcee_sampler
from LyaCosmoParams.data import data_MPGADGET
from LyaCosmoParams.emulator import p1d_arxiv


rootdir="/media/chris/Hard/Work/EmulatorChains/chains" ## Directory where chains are stored

# specify simulation to use to generate synthetic data
test_sim_label=10
if type(test_sim_label)==int:
    drop_sim_number=test_sim_label
    print('will drop sim number {} from emulator'.format(drop_sim_number))
else:
    drop_sim_number=None

# specify simulation suite and P1D mesurements
basedir="/p1d_emulator/sim_suites/Australia20/"
skewers_label='Ns500_wM0.05'
p1d_label=None
z_max=4.0
data=data_MPGADGET.P1D_MPGADGET(basedir=basedir,
                                skewers_label=skewers_label,
                                sim_label=test_sim_label,
                                zmax=z_max)

# Set up emulator training set
z_max=4
arxiv=p1d_arxiv.ArxivP1D(basedir=basedir,drop_sim_number=drop_sim_number,
                            drop_tau_rescalings=True,z_max=z_max,
                            drop_temp_rescalings=True,skewers_label=skewers_label)

## Build emulator
paramList=['mF', 'sigT_Mpc', 'gamma', 'kF_Mpc', 'Delta2_p', 'n_p']
# specify k range
kmax_Mpc=8
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,z_max=z_max,
                                verbose=False,paramList=paramList,train=True,
                                asymmetric_kernel=True,rbf_only=True,
                                emu_type="k_bin",passArxiv=arxiv,
                                kmax_Mpc=kmax_Mpc)

## Set up likelihood object
free_param_names=["As","ns"]
free_param_limits=[[1.1e-09, 3.19e-09], [0.89, 1.05]]

prior=None ## None for uniform prior, otherwise this value sets the width of the Gaussian within the unit prior volume
like=likelihood.Likelihood(data=data,emulator=emu,
                            free_param_names=free_param_names,
                            free_param_limits=free_param_limits,
                            prior_Gauss_rms=prior)

## Set up sampler
sampler = emcee_sampler.EmceeSampler(like=like,rootdir=rootdir,
                        free_param_names=free_param_names,verbose=False,
                        nwalkers=10,
                        save_chain=True,
                        progress=False)

## Cannot call self.log_prob using multiprocess.pool
def log_prob(theta):
    return sampler.like.log_prob(theta)

start = time.time()
sampler.like.go_loud()
sampler.run_sampler(100,1000,log_prob,parallel=True)
end = time.time()
multi_time = end - start
print("Sampling took {0:.1f} seconds".format(multi_time))

