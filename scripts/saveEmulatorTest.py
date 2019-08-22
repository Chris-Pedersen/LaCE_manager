import gp_emulator
import matplotlib.pyplot as plt
import numpy as np
import p1d_arxiv
import matplotlib as mpl
import os

'''
Script to test the saving and loading of trained emulators
'''

assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
repo=os.environ['LYA_EMU_REPO']

## Using the IGM only emulator
basedir='/home/chris/Projects/LyaCosmoParams/p1d_emulator/sim_suites/emulator_256_15072019'
#basedir=repo+"/p1d_emulator/sim_suites/emulator_512_18062019"

paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]


emu_kGP=gp_emulator.GPEmulator(basedir=basedir,
                                   kmax_Mpc=8.0,train=True,
                                   paramList=paramList,skewers_label='Ns256_wM0.05',
                                   emu_type="k_bin",drop_temp_rescalings=False,
                                   drop_tau_rescalings=False,verbose=True)

emu_kGP.saveEmulator()

emu_poly=gp_emulator.GPEmulator(basedir=basedir,
                                   kmax_Mpc=8.0,train=True,
                                   paramList=paramList,skewers_label='Ns256_wM0.05',
                                   emu_type="polyfit",drop_temp_rescalings=False,
                                   drop_tau_rescalings=False,verbose=True)

emu_poly.saveEmulator()