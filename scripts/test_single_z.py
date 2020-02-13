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

repo=os.environ['LYA_EMU_REPO']
basedir=repo+"/p1d_emulator/sim_suites/emulator_512_18062019"
archive=p1d_arxiv.ArxivP1D(basedir=basedir,
                    drop_tau_rescalings=False,
                    drop_temp_rescalings=False)

## Remove all redshifts except 4.0
removes=[]
for aa,item in enumerate(archive.data):
    if item["z"] != 3.0:
        removes.append(aa)

for aa in sorted(removes, reverse=True):
    #del archive.data[np.random.randint(len(archive.data))]
    del archive.data[aa]

print(np.shape(archive.data))
'''
for aa in range(3200):
    del archive.data[np.random.randint(len(archive.data))]
'''
paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]
kmax_Mpc=8

emu=gp_emulator.GPEmulator(basedir,passArxiv=archive,kmax_Mpc=kmax_Mpc,
                               verbose=False,paramList=paramList,train=True,emu_type="polyfit")
testModel={}
for param in paramList:
    testModel[param]=archive.data[0][param]

print(emu.emulate_p1d_Mpc(testModel,emu.training_k_bins))