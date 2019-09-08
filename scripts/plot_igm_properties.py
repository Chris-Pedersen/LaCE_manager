import p1d_arxiv
import numpy as np
import matplotlib.pyplot as plt
import mean_flux_model
from scipy.optimize import curve_fit
import os
import recons_cosmo
import fit_linP

repo=os.environ['LYA_EMU_REPO']
basedir=repo+"/p1d_emulator/sim_suites/emulator_512_18062019"
basedir=repo+"/p1d_emulator/sim_suites/emulator_1024_21062019"
skewers_label='Ns512_wM0.05'


MF=mean_flux_model.MeanFluxModel()
archive=p1d_arxiv.ArxivP1D(basedir=basedir,pick_sim_number=3,
                            drop_tau_rescalings=True,
                            drop_temp_rescalings=True,skewers_label=skewers_label)

data=np.empty([len(archive.data),5])

aa=0
for entry in archive.data:
    IGM_stuff=np.array([entry["z"],
                        entry["mF"],
                        entry["T0"],
                        entry["gamma"],
                        entry["kF_Mpc"]])
    data[aa]=IGM_stuff
    aa+=1

zs=data[:,0]
T0=data[:,2]
gamma=data[:,3]
mF=data[:,1]
kF_Mpc=data[:,4]
kF_kms=np.empty(len(kF_Mpc))

## Import cosmology object to get Mpc -> kms conversion factor
cosmo=recons_cosmo.ReconstructedCosmology(np.flip(zs))

## Convert kF_Mpc into kF_kms
for aa in range(len(zs)):
    ## Iterate backwards..
    conversion_factor=cosmo.reconstruct_Hubble_iz(aa,cosmo.linP_model_fid)/(1+zs[-aa])
    kF_kms[-aa]=kF_Mpc[-aa]/conversion_factor

def get_gamma(z,ln_gamma_0,ln_gamma_1):
    """gamma at the input redshift"""
    xz=np.log((1+z)/(1+3.6))
    ln_gamma_poly=np.poly1d([ln_gamma_0,ln_gamma_1])
    ln_gamma=ln_gamma_poly(xz)
    return np.exp(ln_gamma)

def get_T0(z,a,b,c):
    pivot=3.6
    out=np.empty(len(z))
    for aa in range(len(z)):
        lnz=np.log((1+z[aa])/(1+pivot))
        if z[aa]<pivot:
            log_poly=np.poly1d([a,b])
            ln_f=log_poly(lnz)
            out[aa]=ln_f
        else:
            log_poly=np.poly1d([c,b])
            ln_f=log_poly(lnz)
            out[aa]=ln_f
    return np.exp(out)

def get_mean_flux(z,ln_tau_0,ln_tau_1): ## Order is the wrong way round in these
    """Effective optical depth at the input redshift"""
    xz=np.log((1+z)/(1+3.0))
    ln_tau_poly=np.poly1d([ln_tau_0,ln_tau_1])
    ln_tau=ln_tau_poly(xz)
    return np.exp(-np.exp(ln_tau))

def get_kF_kms(z,ln_kF_0,ln_kF_1):
    """Filtering length at the input redshift (in s/km)"""
    xz=np.log((1+z)/(1+3.5))
    ln_kF_poly=np.poly1d([ln_kF_0,ln_kF_1])
    ln_kF=ln_kF_poly(xz)
    return np.exp(ln_kF)

## Get fit params
fit_mF, err_mF=curve_fit(get_mean_flux, zs, mF)
fit_T0, err_T0=curve_fit(get_T0, zs, T0)
fit_gamma, err_gamma=curve_fit(get_gamma, zs, gamma)
fit_kF_kms, err_kF_kms=curve_fit(get_kF_kms, zs, kF_kms)

## Print values
print("ln_tau_0 = ", fit_mF[1])
print("ln_tau_1 = ", fit_mF[0])
print("ln_kF_0 = ", fit_kF_kms[1])
print("ln_kF_1 = ", fit_kF_kms[0])
print("ln_gamma_0 = ", fit_gamma[1])
print("ln_gamma_1 = ", fit_gamma[0])
print("T0_1 = ", fit_T0[0])
print("T0_2 = ", fit_T0[1])
print("T0_3 = ", fit_T0[2])


fit_zs=np.linspace(zs[0],zs[-1],200)

plt.figure(figsize=(8,15))
plt.subplot(4,1,1)
plt.plot(zs,T0,label="Simulation")
plt.plot(fit_zs,get_T0(fit_zs,fit_T0[0],fit_T0[1],fit_T0[2]),label="Model")
plt.ylabel("T0")
plt.legend()

plt.subplot(4,1,2)
plt.plot(zs,gamma)
plt.plot(fit_zs,get_gamma(fit_zs,fit_gamma[0],fit_gamma[1]))
plt.ylabel("gamma")

plt.subplot(4,1,3)
plt.plot(zs,mF)
plt.plot(fit_zs,get_mean_flux(fit_zs,fit_mF[0],fit_mF[1]))
plt.ylabel("<F>")

plt.subplot(4,1,4)
plt.plot(zs,kF_kms)
plt.plot(fit_zs,get_kF_kms(fit_zs,fit_kF_kms[0],fit_kF_kms[1]))
plt.ylabel("kF_kms")
plt.xlabel("z")

plt.tight_layout()
plt.show()