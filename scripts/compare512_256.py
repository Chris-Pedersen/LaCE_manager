import gp_emulator
import matplotlib.pyplot as plt
import numpy as np
import p1d_arxiv
import matplotlib as mpl
import os

'''
Script to test how well a polyfit emulator trained on 256**3 sims can
match 512**3 sim data for the same model
'''

assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
repo=os.environ['LYA_EMU_REPO']

## Using the IGM only emulator
basedir_256=repo+'p1d_emulator/sim_suites/emulator_256_15072019'
basedir_512=repo+"/p1d_emulator/sim_suites/emulator_512_18062019"

archive_512=p1d_arxiv.ArxivP1D(basedir=basedir_512)

paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]

emu_poly=gp_emulator.GPEmulator(basedir=basedir_256,
                                   kmax_Mpc=8.0,train=True,
                                   paramList=paramList,skewers_label='Ns256_wM0.05',
                                   emu_type="k_bin",drop_temp_rescalings=False,
                                   drop_tau_rescalings=False,verbose=False)

selectedModel=archive_512.data[np.random.randint(len(archive_512.data))]

emu_model={}
for par in paramList:
    emu_model[par]=selectedModel[par]

p1d_emu=emu_poly.emulate_p1d_Mpc(emu_model,emu_poly.training_k_bins,return_covar=False)

plt.figure()
plt.semilogx(emu_poly.training_k_bins,p1d_emu,label="emulated 256")
plt.semilogx(selectedModel["k_Mpc"],selectedModel["p1d_Mpc"],label="Data from 512")
plt.legend()
plt.show()