import matplotlib.pyplot as plt
import numpy as np
import os
import json
# our own modules
import gp_emulator
import z_emulator
import p1d_arxiv



sims=np.linspace(0,199,200,dtype=int)

zmax=4



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

## Construct a list of lists for the theoretical and true error
archive_test=p1d_arxiv.ArxivP1D(basedir=basedir,pick_sim_number=1,
                            drop_tau_rescalings=True,z_max=zmax,
                            drop_temp_rescalings=True,skewers_label=skewers_label)

theory_error=[]
true_error=[]
cross_val=[]
for aa in range(len(archive_test.data)):
    theory_error.append([])
    true_error.append([])
    cross_val.append([])

## Set up k bins for emulator call
k_test=archive_test.data[1]["k_Mpc"][1:]
k_test=k_test[k_test<8]


## Loop over each sim
## for each sim, train GP on all the rest
## For each prediction, save the size of the theoretical error
## and the size of the true error (fractional both)

for sim_num in sims:
    print("\n \n Working on sim number %d \n \n" % sim_num)
    ## Set up arxiv for the mock sim
    archive_true=p1d_arxiv.ArxivP1D(basedir=basedir,pick_sim_number=sim_num,
                            drop_tau_rescalings=True,z_max=zmax,
                            drop_temp_rescalings=True,skewers_label=skewers_label)


    ## Set up emulator trained on all other sims
    emu_archive=p1d_arxiv.ArxivP1D(basedir=basedir,drop_sim_number=sim_num,
                            drop_tau_rescalings=True,z_max=zmax,
                            drop_temp_rescalings=True,skewers_label=skewers_label)

    emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=zmax,
                                verbose=False,paramList=paramList,
                                passArxiv=emu_archive,train=True,
                                emu_type="k_bin")


    ## Loop over each redshift
    for aa,p1d in enumerate(archive_true.data):
        emu_call={}
        for param in paramList:
            ## Find true emulator params
            emu_call[param]=p1d[param]
        prediction,err=emu.emulate_p1d_Mpc(emu_call,k_test,True,p1d["z"])
        err=np.sqrt(np.diag(err))
        truth=p1d["p1d_Mpc"][1:(len(k_test)+1)]

        ## Theoretical error
        theory_error[aa].append((prediction/truth).tolist())

        ## True error
        true_error[aa].append((np.abs(prediction-truth)/truth).tolist())

        cross_val[aa].append(((prediction-truth)/err).tolist())
    
## true_ and theory_error are now lists of lists of arrays
## first index represents the redshift
## second index represents the k bin

'''
## Save these as jsons
with open("/home/chris/Projects/LyaCosmoParams/p1d_emulator/cross_validations/theory_error_full.json","w") as json_file:  
        json.dump(theory_error,json_file)

with open("/home/chris/Projects/LyaCosmoParams/p1d_emulator/cross_validations/true_error_full.json","w") as json_file:  
        json.dump(true_error,json_file)

'''

with open("/home/chris/Projects/LyaCosmoParams/p1d_emulator/cross_validations/cross_val_z.json","w") as json_file:  
        json.dump(cross_val,json_file)

