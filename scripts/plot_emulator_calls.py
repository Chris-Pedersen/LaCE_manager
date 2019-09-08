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
import p1d_arxiv

# read P1D measurement
data=data_PD2013.P1D_PD2013(blind_data=True,zmin=2.1,zmax=4.7)
zs=data.z

#basedir='/home/chris/Projects/LyaCosmoParams/p1d_emulator/sim_suites/emulator_512_18062019/'
basedir='/home/chris/Projects/LyaCosmoParams/p1d_emulator/sim_suites/emulator_256_15072019'
p1d_label=None
skewers_label='Ns256_wM0.05'
undersample_z=1
paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]
max_arxiv_size=None
kmax_Mpc=8.0

emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,kmax_Mpc=kmax_Mpc,
                               undersample_z=undersample_z,max_arxiv_size=max_arxiv_size,
                               verbose=False,paramList=paramList,train=True)

free_parameters=['ln_tau_0','ln_tau_1','ln_gamma_0','ln_kF_0','T0_1','T0_2','T0_3']

sampler = emcee_sampler.EmceeSampler(emulator=emu,free_parameters=free_parameters,verbose=True,
                                priors="Gaussian")


## Parameters we want along x y z axes
param_1="mF"
param_2="kF_Mpc"
param_3="Delta2_p"

## Each sampler position will return 12 models, for the 12 redshift bins
## and we have 70 initial sampling positions
## so we will be plotting ~800 models.
## Lets try it

likes_1=np.array([])
likes_2=np.array([])
likes_3=np.array([])

for samplerPositions in sampler.p0:
    #print(samplerPositions)
    paramSet=sampler.like.theory.get_emulator_calls(sampler.like.parameters_from_sampling_point(samplerPositions))
    for emu_call in paramSet:
        #print("New call:")
        #print(emu_call["mF"])
        likes_1=np.append(likes_1,emu_call[param_1])
        likes_2=np.append(likes_2,emu_call[param_2])
        likes_3=np.append(likes_3,emu_call[param_3])

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(likes_1, likes_2, likes_3,s=8)
ax.set_xlabel(param_1)
ax.set_ylabel(param_2)
ax.set_zlabel(param_3)


###
## This bit plots the training points
###

param_1="mF"
param_2="kF_Mpc"
param_3="Delta2_p"
## Get archive data
emu_data=emu.arxiv.data
Nemu=len(emu_data)

# figure out values of param_1,param_2 in arxiv
emu_1=np.array([emu_data[i][param_1]for i in range(Nemu)])
emu_2=np.array([emu_data[i][param_2]for i in range(Nemu)])
emu_3=np.array([emu_data[i][param_3]for i in range(Nemu)])

emu_z=np.array([emu_data[i]['z']for i in range(Nemu)])

zmin=min(emu_z)
zmax=max(emu_z)
fig = plt.figure()
ax = plt.axes(projection="3d")
#ax.scatter3D(emu_1, emu_2, emu_3, c=emu_z, cmap='brg',s=8,marker="X")
ax.scatter3D(emu_1, emu_2, emu_3,s=8,color="red",label="Training points")
ax.scatter3D(likes_1, likes_2, likes_3,s=8,label="Emulator calls",color="black")
ax.set_xlabel(param_1)
ax.set_ylabel(param_2)
ax.set_zlabel(param_3)
ax.legend()
plt.show()
