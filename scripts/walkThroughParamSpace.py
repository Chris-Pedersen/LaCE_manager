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
numPairs=50
k=1.5

repo=os.environ['LYA_EMU_REPO']
basedir_512=repo+'/p1d_emulator/sim_suites/emulator_512_18062019/'
basedir=repo+'/p1d_emulator/sim_suites/emulator_256_15072019'

## IGM only emulator
paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]

## Initialise emulators
emu_no_rescalings=gp_emulator.GPEmulator(basedir=basedir,paramList=paramList,drop_tau_rescalings=True,
                                drop_temp_rescalings=True,emu_type="polyfit",skewers_label='Ns256_wM0.05',
                                kmax_Mpc=8.0,
                                train=True)
emu_no_rescalings.saveEmulator()
emu_with_rescalings=gp_emulator.GPEmulator(basedir=basedir,paramList=paramList, emu_type="polyfit",
                                skewers_label='Ns256_wM0.05',kmax_Mpc=8.0,
                                train=True,verbose=False)
emu_with_rescalings.saveEmulator()

def get_walk():
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
        return fracError,fracError_res

paths=np.empty((numPairs,numPoints))
paths_res=np.empty((numPairs,numPoints))

for aa in range(numPairs):
      paths[aa],paths_res[aa]=get_walk()  

paths_512=np.empty((numPairs,numPoints))
paths_res_512=np.empty((numPairs,numPoints))

## Initialise emulators
emu_no_rescalings=gp_emulator.GPEmulator(basedir=basedir_512,paramList=paramList,drop_tau_rescalings=True,
                                drop_temp_rescalings=True,emu_type="polyfit",
                                kmax_Mpc=8.0,
                                train=True)
emu_no_rescalings.saveEmulator()
emu_with_rescalings=gp_emulator.GPEmulator(basedir=basedir_512,paramList=paramList, emu_type="polyfit",
                                kmax_Mpc=8.0,
                                train=True,verbose=False)

for aa in range(numPairs):
      paths_512[aa],paths_res_512[aa]=get_walk()  

plt.figure()
#plt.errorbar(np.linspace(0,numPoints-1,numPoints),np.mean(paths,axis=0),
#        yerr=np.std(paths,axis=0),label="No rescalings")
#plt.errorbar(np.linspace(0,numPoints-1,numPoints),np.mean(paths_res,axis=0),
#        yerr=np.std(paths_res,axis=0),label="With rescalings")
plt.plot(np.linspace(0,numPoints-1,numPoints),np.mean(paths,axis=0),label="256 emulator, no rescalings",color="C0")
plt.plot(np.linspace(0,numPoints-1,numPoints),np.mean(paths_res,axis=0),label="256 emulator, with rescalings",color="C0",linestyle="dashed")
plt.plot(np.linspace(0,numPoints-1,numPoints),np.mean(paths_512,axis=0),label="512 emulator, no rescalings",color="C1")
plt.plot(np.linspace(0,numPoints-1,numPoints),np.mean(paths_res_512,axis=0),label="512 emulator, with rescalings",color="C1",linestyle="dashed")
plt.ylabel("Fractional error")
plt.xlabel("Step number")
plt.legend()
plt.show()    