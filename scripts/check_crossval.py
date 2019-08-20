import gp_emulator
import matplotlib.pyplot as plt
import numpy as np

basedir='/home/chris/Projects/LyaCosmoParams/p1d_emulator/sim_suites/emulator_512_18062019'

paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]

emu_high=gp_emulator.GPEmulator(basedir=basedir,undersample_z=1,
                                   kmax_Mpc=10,train=False,max_arxiv_size=2000,
                                   drop_tau_rescalings=False,paramList=paramList,
                                   drop_temp_rescalings=False,emu_type="polyfit",
                                   set_noise_var=1e-2)

#print(emu_kGP.gp.param_array)
'''
emu_low=gp_emulator.GPEmulator(basedir=basedir,undersample_z=1,
                                   kmax_Mpc=10,train=False,max_arxiv_size=2000,
                                   drop_tau_rescalings=False,paramList=paramList,
                                   drop_temp_rescalings=False,emu_type="k_bin",
                                   passArxiv=emu_high.arxiv,set_noise_var=1e-10)
'''
emu_high.crossValidation()
#emu_low.crossValidation()
'''
emu_no_cv=gp_emulator.GPEmulator(basedir=basedir,undersample_z=1,
                                   kmax_Mpc=10,train=True,max_arxiv_size=200,
                                   drop_tau_rescalings=False,paramList=paramList,
                                   drop_temp_rescalings=False,emu_type="k_bin",
                                   set_noise_var=1e-2)


## construct model dictionaries for these two points
testModel = {}
for name in paramList:
    testModel[name]=emu_high.arxiv.data[7][name]

## Move model slightly away from a training point
testModel["sigT_Mpc"]=testModel["sigT_Mpc"]*0.9
testModel["mF"]=testModel["mF"]*1.005

k=emu_high.arxiv.data[0]["k_Mpc"][1:35]

emu_high.crossValidation()

y,cov=emu_high.emulate_p1d_Mpc(testModel,k,return_covar=True)
'''




