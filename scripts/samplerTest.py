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
import z_emulator


# read P1D measurement
z_list=np.array([2.0,2.75,3.25,4.0])
data=data_MPGADGET.P1D_MPGADGET(sim_number=17)
zs=data.z



repo=os.environ['LYA_EMU_REPO']
skewers_label='Ns256_wM0.05'
#skewers_label=None
basedir=repo+"/p1d_emulator/sim_suites/emulator_256_28082019/"
#basedir=repo+"/p1d_emulator/sim_suites/emulator_256_15072019/"
p1d_label=None
undersample_z=1
paramList=["mF","sigT_Mpc","gamma","kF_Mpc","Delta2_p"]
max_arxiv_size=None
kmax_Mpc=8


'''
emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="polyfit",z_list=z_list,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)

emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="polyfit",z_list=z_list,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)


theory=lya_theory.LyaTheory(zs,emulator=emu)

free_param_names=['ln_tau_0','ln_tau_1','ln_gamma_0','T0_1','T0_2','T0_3']

like=likelihood.Likelihood(data=data,theory=theory,
                            free_param_names=free_param_names,verbose=False,
                            prior_Gauss_rms=0.15)

sampler = emcee_sampler.EmceeSampler(like=like,emulator=emu,
                        free_param_names=free_param_names,verbose=True,
                        nwalkers=100)


for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)


sampler.like.go_silent()
sampler.store_distances=True
sampler.run_burn_in(nsteps=50)
#sampler.run_chains(nsteps=200)

plt.figure()
for aa,array in enumerate(sampler.distances):
    plt.hist(array,label="z=%.2f" % sampler.like.theory.zs[aa],alpha=0.5,bins=300)

plt.xlabel("Euclidean distance to nearest training point")
plt.ylabel("Counts")
plt.legend()
plt.show()


sampler.run_chains(nsteps=50)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.sampler.acceptance_fraction)))

sampler.plot_corner(cube=True,mock_values=True)
#sampler.plot_best_fit()
'''
