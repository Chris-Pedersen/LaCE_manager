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
from getdist import plots, MCSamples

test_sim_number=50

# read P1D measurement
z_list=np.array([2.0,2.75,3.25,4.0])
data=data_MPGADGET.P1D_MPGADGET(sim_number=test_sim_number)
zs=data.z

repo=os.environ['LYA_EMU_REPO']
skewers_label='Ns256_wM0.05'
#skewers_label=None
basedir=repo+"/p1d_emulator/sim_suites/emulator_256_28082019/"
#basedir=repo+"/p1d_emulator/sim_suites/emulator_256_15072019/"
p1d_label=None
undersample_z=1
paramList=["mF","sigT_Mpc","gamma","kF_Mpc","Delta2_p","n_p"]
max_arxiv_size=None
kmax_Mpc=8

archive=p1d_arxiv.ArxivP1D(basedir=basedir,
                            drop_tau_rescalings=True,z_max=4,
                            drop_sim_number=test_sim_number,
                            drop_temp_rescalings=True,skewers_label=skewers_label)


emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",passArxiv=archive,checkHulls=False)
'''
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                passArxiv=archive,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",z_list=z_list, checkHulls=False,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)
'''

free_parameters=['mF',"Delta2_p","sigT_Mpc","gamma","kF_Mpc","n_p"]

like=likelihood.simpleLikelihood(data=data,emulator=emu,
                            free_parameters=free_parameters,verbose=False,
                            prior_Gauss_rms=0.15)

#like.plot_p1d()

sampler = emcee_sampler.EmceeSampler(like=like,
                        free_parameters=free_parameters,verbose=True,
                        nwalkers=20)

sampler.read_chain_from_file("test_chain")
#sampler.plot_best_fit()
sampler.plot_corner(mock_values=True)


g = plots.get_subplot_plotter()
g.triangle_plot([samples], filled=True)