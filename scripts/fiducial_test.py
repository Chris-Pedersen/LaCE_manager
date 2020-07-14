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
import p1d_arxiv

'''
Script to check how close our fiducial models for each mock data
sim actually reproduce the mock data
'''

# read P1D measurement
z_list=np.array([2.0,2.75,3.25,3.5,4.0])
data=data_MPGADGET.P1D_MPGADGET(z_list=z_list,filename="256_mock_199.json")
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
skewers_label='Ns256_wM0.05'
#skewers_label=None
basedir=repo+"/p1d_emulator/sim_suites/emulator_256_28082019/"
#basedir=repo+"/p1d_emulator/sim_suites/emulator_256_15072019/"
p1d_label=None
undersample_z=1
paramList=["mF","sigT_Mpc","gamma","kF_Mpc","Delta2_p"]
max_arxiv_size=None
kmax_Mpc=8

archive=p1d_arxiv.ArxivP1D(basedir=basedir,nsamples=199,
                            drop_tau_rescalings=True,z_max=4,
                            drop_temp_rescalings=True,skewers_label=skewers_label)
'''
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",z_list=z_list,passArxiv=archive,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)
'''
emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",z_list=z_list,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)

theory=lya_theory.LyaTheory(zs,emulator=emu,T_model_fid=thermal_model,
                                            kF_model_fid=kF_model,
                                            mf_model_fid=mf_model)


free_param_names=['ln_tau_0','ln_tau_1']#,'ln_gamma_0','T0_1','T0_2','T0_3']

like=likelihood.Likelihood(data=data,theory=theory,
                            free_param_names=free_param_names,verbose=False,
                            prior_Gauss_rms=0.15,min_kp_kms=0.0041)

sampler = emcee_sampler.EmceeSampler(like=like,emulator=emu,
                        free_param_names=free_param_names,verbose=True,
                        nwalkers=100)

like.plot_p1d()
