import numpy as np
import matplotlib.pyplot as plt
import time
# our modules
import data_MPGADGET
import p1d_arxiv
import gp_emulator
import likelihood

# specify simulation suite and P1D mesurements
basedir="/p1d_emulator/sim_suites/Australia20/"
skewers_label='Ns500_wM0.05'
p1d_label=None
# specify simulation to use to generate synthetic data
test_sim_num=15
data=data_MPGADGET.P1D_MPGADGET(sim_number=test_sim_num,
        basedir=basedir,skewers_label=skewers_label,data_cov_factor=1)

# specify redshift range
z_max=4
# do not use test_sim_num that was used in generating mock data
arxiv=p1d_arxiv.ArxivP1D(basedir=basedir,drop_sim_number=test_sim_num,
        drop_tau_rescalings=True,z_max=z_max,
        drop_temp_rescalings=True,skewers_label=skewers_label)

# specify parameters to be used in emulator
paramList=['mF', 'sigT_Mpc', 'gamma', 'kF_Mpc', 'Delta2_p', 'n_p']
# specify k range
kmax_Mpc=8
# setup GP emulator
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,z_max=z_max,
        verbose=True,paramList=paramList,train=False,emu_type="k_bin",
        passArxiv=arxiv,kmax_Mpc=kmax_Mpc)
emu.load_default()

# Likelihood parameters
add_z_evol=False
if add_z_evol:
    like_params=["Delta2_star","n_star","ln_tau_0","ln_tau_1","ln_sigT_kms_0","ln_sigT_kms_1","ln_gamma_0","ln_gamma_1"]
    like_param_limits=[[0.24, 0.47], [-2.352, -2.25], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]]
else:
    like_params=["Delta2_star","n_star","ln_tau_0","ln_sigT_kms_0","ln_gamma_0"]
    like_param_limits=[[0.24, 0.47], [-2.352, -2.25], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]]

like=likelihood.Likelihood(data=data,emulator=emu,
        free_parameters=like_params,free_param_limits=like_param_limits,
        verbose=True,prior_Gauss_rms=-1,emu_cov_factor=1,
        use_sim_cosmo=False)
like.go_loud()

## Evaluate log_prob at random point in parameter space
theta=np.ones(len(like_params))*0.5 ## Just pick the middle of likelihood space
print('starting point')
for par in like.parameters_from_sampling_point(theta):
    print(par.info_str())
tic=time.perf_counter()
chi2_test=like.get_chi2(theta)
toc=time.perf_counter()
print('chi2 test',chi2_test)
print("Took ", toc-tic, "s to evaluate a likelihood")

from scipy.optimize import minimize
print('minimize chi2')
results=minimize(like.get_chi2, x0=theta)
print('minimum chi2 =',results.fun)
for par in like.parameters_from_sampling_point(results.x):
    print(par.info_str())
