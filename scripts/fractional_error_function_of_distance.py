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

n_points=10000
mF_res=False
temp_res=False
# read P1D measurement
data=data_MPGADGET.P1D_MPGADGET()
zs=data.z

k_mpc=data.k*80

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

emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)


k_point=k_mpc[5]

plt.figure()
for aa, emulator in enumerate(emu.emulators):
    ## Set up min/max prior volume
    limits={}
    for param in paramList:
        par_values=emulator.arxiv.get_param_values(param)
        limits[param]=np.array([min(par_values),max(par_values)])
    ## Randomly sample prior volume and get fractional error
    pred_dict={}
    distances=np.empty(n_points)
    frac_error=np.empty(n_points)
    for bb in range(n_points):
        for param in paramList:
            pred_dict[param]=np.random.uniform(limits[param][0],limits[param][1])
        distances[bb]=emulator.get_nearest_distance(pred_dict)
        p1d,error=emulator.emulate_p1d_Mpc(pred_dict,k_point,return_covar=True)
        inside_hull=emulator.check_in_hull(pred_dict)
        print(inside_hull)
        frac_error[bb]=error/p1d
    sigma_rbf=emulator.gp.param_array[1]
    sigma_linear=emulator.gp.param_array[0]
    lengthscale=emulator.gp.param_array[2]
    ## Plot scatter
    plt.subplot(2,1,1)
    plt.scatter(distances,frac_error,s=1.5,label=r"z=%.2f, $\sigma^2_\mathrm{RBF}=%.3f$, $\sigma^2_\mathrm{linear}=%.3f$, $l_\mathrm{RBF}=%.3f$" % (emu.zs[aa],sigma_rbf,sigma_linear,lengthscale))
    plt.subplot(2,1,2)
    plt.hist(distances,bins=100,label="z%.2f"% emu.zs[aa],alpha=0.35)

plt.subplot(2,1,1)
plt.title(r"$\bar{F}$ rescalings = %s, temp rescalings = %s" % (mF_res, temp_res))
plt.ylabel("Fractional error")
plt.legend(loc="upper left",markerscale=2.5)
plt.subplot(2,1,2)
plt.xlabel("Euclidean distance to nearest training point")
plt.tight_layout()
plt.legend()
plt.show()