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
z_list=np.array([2.0,3.0,4.0])
data=data_MPGADGET.P1D_MPGADGET(z_list=z_list,filename="1024_mock_1.json")
zs=data.z

tau_values=[data.like_params["ln_tau_1"],data.like_params["ln_tau_0"]]
gamma_values=[data.like_params["ln_gamma_1"],data.like_params["ln_gamma_0"]]
T0_values=[data.like_params["T0_1"],data.like_params["T0_2"],data.like_params["T0_3"]]
kF_values=[data.like_params["ln_kF_1"],data.like_params["ln_kF_0"]]


mf_model=mean_flux_model.MeanFluxModel(ln_tau_coeff=tau_values)
thermal_model=thermal_model.ThermalModel(ln_gamma_coeff=gamma_values,
                                ln_T0_coeff=T0_values)
kF_model=pressure_model.PressureModel(ln_kF_coeff=kF_values)

repo=os.environ['LYA_EMU_REPO']
#skewers_label='Ns256_wM0.05'
skewers_label=None
basedir=repo+"/p1d_emulator/sim_suites/emulator_512_18062019/"
#basedir=repo+"/p1d_emulator/sim_suites/emulator_256_15072019/"
p1d_label=None
undersample_z=1
paramList=["mF","sigT_Mpc","gamma","kF_Mpc"]
max_arxiv_size=None
kmax_Mpc=8

#emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
#                               undersample_z=undersample_z,max_arxiv_size=max_arxiv_size,z_max=4,
#                               verbose=False,paramList=paramList,train=True,emu_type="polyfit")

emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="polyfit",z_list=z_list,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)
#emu.saveEmulator()


theory=lya_theory.LyaTheory(zs,emulator=emu,T_model_fid=thermal_model,
                                            kF_model_fid=kF_model,
                                            mf_model_fid=mf_model)

free_parameters=['ln_tau_0','ln_tau_1']#,'ln_gamma_0','T0_1','T0_2','T0_3']

like=likelihood.Likelihood(data=data,theory=theory,
                            free_parameters=free_parameters,verbose=False,
                            prior_Gauss_rms=0.15)

sampler = emcee_sampler.EmceeSampler(like=like,emulator=emu,
                        free_parameters=free_parameters,verbose=True,
                        nwalkers=100)


for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)


sampler.like.go_silent()
sampler.run_burn_in(nsteps=50)
sampler.run_chains(nsteps=50)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.sampler.acceptance_fraction)))

sampler.plot_corner(cube=True,mock_values=True)
sampler.plot_best_fit()
