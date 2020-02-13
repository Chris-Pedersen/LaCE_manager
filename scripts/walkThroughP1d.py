import p1d_arxiv
import gp_emulator
import os
import numpy as np
import matplotlib.pyplot as plt

'''
Script to pick two random points in parameter space
and walk between them plotting the fractional error
along the way, with and without temperature rescalings
'''

numPoints=100

repo=os.environ['LYA_EMU_REPO']
#basedir=repo+'/p1d_emulator/sim_suites/emulator_512_18062019/'
basedir=repo+'/p1d_emulator/sim_suites/emulator_256_15072019'

## IGM only emulator
paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]

## Initialise emulators
emu=gp_emulator.GPEmulator(basedir=basedir,paramList=paramList,drop_tau_rescalings=True,
                                drop_temp_rescalings=True,emu_type="polyfit",skewers_label='Ns256_wM0.05',
                                kmax_Mpc=8.0,
                                train=True)

## Set a range of k values to emulate
k=emu.training_k_bins[1:]


## Pick 2 random points in parameter space
p1=np.random.randint(len(emu_no_rescalings.arxiv.data))
p2=np.random.randint(len(emu_no_rescalings.arxiv.data))
point1={}
point2={}

for par in paramList:
        point1[par]=emu_no_rescalings.arxiv.data[p1][par]
        point2[par]=emu_no_rescalings.arxiv.data[p2][par]

delta_pars={}
for par in paramList:
    delta_pars[par]=(point2[par]-point1[par])/numPoints

stepModel=dict(point1)
fracError=np.empty(numPoints)
fracError_res=np.empty(numPoints)
prediction,error=emu_no_rescalings.emulate_p1d_Mpc(point1,k,return_covar=True)
fracError[0]=np.sqrt(error)/prediction
prediction,error=emu_with_rescalings.emulate_p1d_Mpc(point1,k,return_covar=True)
fracError_res[0]=np.sqrt(error)/prediction

for point in range(1,numPoints):
        for par in paramList:
                stepModel[par]=stepModel[par]+delta_pars[par]
                prediction,error=emu_no_rescalings.emulate_p1d_Mpc(stepModel,k,return_covar=True)
                fracError[point]=np.sqrt(error)/prediction
                prediction,error=emu_with_rescalings.emulate_p1d_Mpc(stepModel,k,return_covar=True)
                fracError_res[point]=np.sqrt(error)/prediction


## Want a 3d plot with:
## x axis being k
## y axis being steps in parameter space
## z axis being P_1d(k)