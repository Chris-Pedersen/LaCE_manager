import data_PD2013
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import p1d_arxiv
from scipy.optimize import curve_fit

'''
Script to take a chosen simulation in the emulator archive
and save a json dictionary of the format used by the data_MPGADGET
class, including the best fit likelihood parameters.
'''

sim_num=0

## Pick an emulator suite and a simulation number
repo=os.environ['LYA_EMU_REPO']
basedir=repo+"/p1d_emulator/sim_suites/emulator_1024_21062019"
skewers_label='Ns512_wM0.05'
archive=p1d_arxiv.ArxivP1D(basedir=basedir,pick_sim_number=sim_num,
                            drop_tau_rescalings=True,z_max=4,
                            drop_temp_rescalings=True,skewers_label=skewers_label)
sim_data=archive.data

## Save the P1D values
for item in (sim_data):
    if item["z"]==2.0 or item["z"]==3.0 or item["z"]==4.0:
        print("Sim z=",item["z"])
        print("mF=",item["mF"])
        print("T0=",item["T0"])
        print("gamma=",item["gamma"])
        print("kF_Mpc",item["kF_Mpc"])
        print("Delta2_p=",item["Delta2_p"])
        print("\n")
