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
paramList=["mF","sigT_Mpc","gamma","kF_Mpc","Delta2_p","n_p"]
max_arxiv_size=None
kmax_Mpc=8

emulator=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="polyfit",checkHulls=False,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)


k_point=k_mpc[5]

plt.figure()
## Set up min/max prior volume
limits={}
for param in paramList:
    par_values=emulator.arxiv.get_param_values(param)
    limits[param]=np.array([min(par_values),max(par_values)])
## Randomly sample prior volume and get fractional error
pred_dict={}
distances=np.empty(n_points)
frac_error=np.empty(n_points)
inhull=np.empty(n_points,dtype=bool)
for bb in range(n_points):
    for param in paramList:
        pred_dict[param]=np.random.uniform(limits[param][0],limits[param][1])
    distances[bb]=emulator.get_nearest_distance(pred_dict)
    p1d,error=emulator.emulate_p1d_Mpc(pred_dict,k_point,return_covar=True)
    inside_hull=emulator.check_in_hull(pred_dict)
    inhull[bb]=inside_hull
    frac_error[bb]=error/p1d
sigma_rbf=emulator.gp.param_array[1]
sigma_linear=emulator.gp.param_array[0]
lengthscale=emulator.gp.param_array[2]
## Plot scatter
plt.scatter(distances[np.invert(inhull)],frac_error[np.invert(inhull)],s=1.5,label=r"$\sigma^2_\mathrm{RBF}=%.3f$, $\sigma^2_\mathrm{linear}=%.3f$, $l_\mathrm{RBF}=%.3f$" % (sigma_rbf,sigma_linear,lengthscale))
plt.scatter(distances[inhull],frac_error[inhull],s=1.5,label="+ outside hull")
plt.title(r"$\bar{F}$ rescalings = %s, temp rescalings = %s" % (mF_res, temp_res))
plt.ylabel("Fractional error")
plt.legend(loc="upper left",markerscale=2.5)
plt.xlabel("Euclidean distance to nearest training point")
plt.tight_layout()
plt.show()

'''
theory=lya_theory.LyaTheory(zs,emulator=emulator,T_model_fid=thermal_model,
                                            kF_model_fid=kF_model,
                                            mf_model_fid=mf_model)

free_param_names=['ln_tau_0','ln_tau_1','ln_gamma_0','T0_1','T0_2','T0_3']

like=likelihood.Likelihood(data=data,theory=theory,
                            free_param_names=free_param_names,verbose=False,
                            prior_Gauss_rms=0.15)

sampler = emcee_sampler.EmceeSampler(like=like,emulator=emulator,
                        free_param_names=free_param_names,verbose=True,
                        nwalkers=100)


for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)


sampler.like.go_silent()
sampler.store_distances=True
sampler.run_burn_in(nsteps=50)
#sampler.run_chains(nsteps=200)

plt.figure()
plt.hist(np.ndarray.flatten(np.asarray(sampler.distances)),bins=200)
plt.xlabel("Euclidean distance to nearest training point")
plt.ylabel("Counts")
plt.legend()
plt.show()
'''
