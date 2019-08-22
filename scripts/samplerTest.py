import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import emcee
import corner
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


# read P1D measurement
data=data_MPGADGET.P1D_MPGADGET(z_list=np.array([2,3,4]))
zs=data.z
repo=os.environ['LYA_EMU_REPO']
basedir=repo+"/p1d_emulator/sim_suites/emulator_512_18062019/"
p1d_label=None
skewers_label=None
undersample_z=1
paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]
max_arxiv_size=None
kmax_Mpc=8

emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                               undersample_z=undersample_z,max_arxiv_size=max_arxiv_size,
                               verbose=False,paramList=paramList,train=True,emu_type="polyfit")

theory=lya_theory.LyaTheory(zs,emulator=emu)


free_parameters=['ln_tau_0','ln_tau_1','ln_gamma_0','ln_kF_0','T0_1','T0_2','T0_3']

like=likelihood.Likelihood(data=data,theory=theory,
                            free_parameters=free_parameters,verbose=False)

sampler = emcee_sampler.EmceeSampler(like=like,emulator=emu,free_parameters=free_parameters,verbose=True)

for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)

sampler.like.go_silent()
sampler.run_burn_in(nsteps=200)
sampler.run_chains(nsteps=1000)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.sampler.acceptance_fraction)))

sampler.plot_histograms(cube=True)
sampler.plot_histograms(cube=False)

