import gp_emulator
import matplotlib.pyplot as plt
import numpy as np

'''
Script to pick a random point within the prior volume
and return the predicted P1D(k) from the k bin and polyfit
emulators for the same model
'''

arxivSize=2000
kMax=8

## Using the IGM only emulator
basedir='/home/chris/Projects/LyaCosmoParams/p1d_emulator/sim_suites/emulator_512_18062019'

paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]

emu_kGP=gp_emulator.GPEmulator(basedir=basedir,undersample_z=1,
                                   kmax_Mpc=kMax,train=True,max_arxiv_size=arxivSize,
                                   drop_tau_rescalings=False,paramList=paramList,
                                   drop_temp_rescalings=False,emu_type="k_bin")

emu_poly=gp_emulator.GPEmulator(basedir=basedir,undersample_z=1,
                                   kmax_Mpc=kMax,train=True,max_arxiv_size=arxivSize,
                                   drop_tau_rescalings=False,paramList=paramList,
                                   drop_temp_rescalings=False,emu_type="polyfit",
                                   passArxiv=emu_kGP.arxiv)

## Initialise model dictionary and find parameter limits
testModel = {}
parameterLimits=emu_kGP.paramLimits
for aa in range(len(paramList)):
    ## For each parameter pick a spot in the prior volume
    testModel[paramList[aa]]=np.random.uniform(parameterLimits[aa][0],parameterLimits[aa][1])

k=np.linspace(emu_kGP.arxiv.data[0]["k_Mpc"][1],emu_kGP.kmax_Mpc-0.2,100)

y,cov=emu_kGP.emulate_p1d_Mpc(testModel,k,return_covar=True)
y_poly,cov_poly=emu_poly.emulate_p1d_Mpc(testModel,k,return_covar=True)

plt.figure()
plt.errorbar(k,y*k,yerr=np.sqrt(np.diag(cov)),label="k bin emulator")
plt.errorbar(k,y_poly*k,yerr=np.sqrt(np.diag(cov_poly)),label="polyfit emulator")
#plt.yscale("log")
plt.ylabel(r"$kP(k)_\mathrm{1D}$")
plt.xlabel(r"$k[1/Mpc]$")
plt.legend()
plt.show()
