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
import gp_emulator


# read P1D measurement
z_list=np.array([2.0,2.5,3.25,4.0])
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

#emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
#                               undersample_z=undersample_z,max_arxiv_size=max_arxiv_size,z_max=4,
#                               verbose=False,paramList=paramList,train=True,emu_type="polyfit")

emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="polyfit",z_list=z_list,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)

'''
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="polyfit",z_list=z_list,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)
#emu.saveEmulator()

'''
theory=lya_theory.LyaTheory(zs,emulator=emu,T_model_fid=thermal_model,
                                            kF_model_fid=kF_model,
                                            mf_model_fid=mf_model)

free_parameters=['ln_tau_0','ln_tau_1','ln_gamma_0','T0_1','T0_2','T0_3']

like=likelihood.Likelihood(data=data,theory=theory,
                            free_parameters=free_parameters,verbose=False,
                            prior_Gauss_rms=0.15)

sampler = emcee_sampler.EmceeSampler(like=like,emulator=emu,
                        free_parameters=free_parameters,verbose=True,
                        nwalkers=100)


'''
fig = plt.figure()
    
ax = plt.axes(projection="3d")
ax.scatter3D(emu.X_param_grid[:,0], emu.X_param_grid[:,1], emu.X_param_grid[:,2])
#ax.scatter3D(emu_1, emu_2, emu_3,s=8,color="red",label="Training points")
#ax.scatter3D(likes_1, likes_2, likes_3,s=8,label="Emulator calls",color="black")
ax.set_xlabel(emu.paramList[0])
ax.set_ylabel(emu.paramList[1])
ax.set_zlabel(emu.paramList[2])
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(emu.X_param_grid[:,2], emu.X_param_grid[:,3], emu.X_param_grid[:,4])
ax.set_xlabel(emu.paramList[2])
ax.set_ylabel(emu.paramList[3])
ax.set_zlabel(emu.paramList[4])
plt.show()
'''



#sampler.like.theory.emulator.emulators[0].arxiv.plot_3D_samples("mF","gamma","sigT_Mpc")
parameter_list=emu.emulators[0].paramList
## Plot unit cube training parameters
for aa,emu_test in enumerate(sampler.like.theory.emulator.emulators):
    ## Are our redshifts correct?
    print("z in training sets:", emu_test.arxiv.data[2]["z"])
    print("z in emulator list:", emu.zs[aa])
    print("z in theory list:", sampler.like.theory.zs[aa])
    likes_1=[]
    likes_2=[]
    likes_3=[]
    likes_4=[]
    likes_5=[]
    for samplerPositions in sampler.p0:
    #print(samplerPositions)
        emu_call=sampler.like.theory.get_emulator_calls(sampler.like.parameters_from_sampling_point(samplerPositions))
        print(len(emu_call)-aa-1)
        unit_params=emu_test.return_unit_call(emu_call[len(emu_call)-aa-1])
        ## Append only values for that redshift
        likes_1=np.append(likes_1,unit_params[0]) 
        likes_2=np.append(likes_2,unit_params[1])
        likes_3=np.append(likes_3,unit_params[2])
        likes_4=np.append(likes_4,unit_params[3])
        likes_5=np.append(likes_5,unit_params[4])

    print("z=",emu_test.arxiv.data[0]["z"])
    emu_test.printPriorVolume()
    fig = plt.figure()
    
    ax = plt.axes(projection="3d")
    ax.scatter3D(emu_test.X_param_grid[:,0], emu_test.X_param_grid[:,1], emu_test.X_param_grid[:,2],label="Training points",color="blue")
    #ax.scatter3D(emu_1, emu_2, emu_3,s=8,color="red",label="Training points")
    ax.scatter3D(likes_1, likes_2, likes_3,s=8,label="Emulator calls",color="red")
    ax.set_xlabel(emu_test.paramList[0])
    ax.set_ylabel(emu_test.paramList[1])
    ax.set_zlabel(emu_test.paramList[2])
    plt.legend()
    plt.title("z=%.2f" % sampler.like.theory.emulator.zs[aa])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(emu_test.X_param_grid[:,2], emu_test.X_param_grid[:,3], emu_test.X_param_grid[:,4],color="blue",label="Training points")
    ax.scatter3D(likes_3, likes_4, likes_5,s=8,label="Emulator calls",color="red")
    ax.set_xlabel(emu_test.paramList[2])
    ax.set_ylabel(emu_test.paramList[3])
    ax.set_zlabel(emu_test.paramList[4])
    plt.title("z=%.2f" % sampler.like.theory.emulator.zs[aa])
    plt.legend()
    #ax.legend()

plt.show()




''' 
## Old
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
'''
