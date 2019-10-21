import data_PD2013
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import p1d_arxiv
from scipy.optimize import curve_fit
import recons_cosmo

'''
Script to take a chosen simulation in the emulator archive
and save a json dictionary of the format used by the data_MPGADGET
class, including the best fit likelihood parameters.
'''

sim_num=3
save=False ## Double check before running this to make sure we don't
           ## accidentally overwrite

## Pick an emulator suite and a simulation number
repo=os.environ['LYA_EMU_REPO']
basedir=repo+"/p1d_emulator/sim_suites/emulator_1024_21062019"
skewers_label='Ns512_wM0.05'
archive=p1d_arxiv.ArxivP1D(basedir=basedir,pick_sim_number=sim_num,
                            drop_tau_rescalings=True,z_max=5,
                            drop_temp_rescalings=True,skewers_label=skewers_label)
sim_data=archive.data
z_sim=np.empty(len(archive.data))

## Get array of redshifts for the sim data
for aa,item in enumerate(sim_data):
    z_sim[aa]=item["z"]

## Import cosmology object to get Mpc -> kms conversion factor
cosmo=recons_cosmo.ReconstructedCosmology(np.flip(z_sim))

saveList=[]
saveDict={}

## Save the P1D values
for aa,item in enumerate(sim_data):
    p1d_Mpc=np.asarray(item["p1d_Mpc"][1:])
    k_Mpc=np.asarray(item["k_Mpc"][1:])
    conversion_factor=cosmo.reconstruct_Hubble_iz(len(z_sim)-aa-1,cosmo.linP_model_fid)/(1+z_sim[aa])
    p1d=p1d_Mpc*conversion_factor
    k=k_Mpc/conversion_factor
    z=item["z"]
    p1d_data={}
    p1d_data["k_kms"]=k.tolist()
    p1d_data["p1d_kms"]=p1d.tolist()
    p1d_data["z"]=z
    print("z=",z)
    print("Conversion factor=",conversion_factor)
    saveList.append(p1d_data)

saveDict["data"]=saveList

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

saveDict["like_params"]={}
saveDict["like_params"]["ln_tau_0"]=fit_mF[1]
saveDict["like_params"]["ln_tau_1"]=fit_mF[0]
saveDict["like_params"]["ln_kF_0"]=fit_kF_kms[1]
saveDict["like_params"]["ln_kF_1"]=fit_kF_kms[0]
saveDict["like_params"]["ln_gamma_0"]=fit_gamma[1]
saveDict["like_params"]["ln_gamma_1"]=fit_gamma[0]
saveDict["like_params"]["T0_1"]=fit_T0[0]
saveDict["like_params"]["T0_2"]=fit_T0[1]
saveDict["like_params"]["T0_3"]=fit_T0[2]

if save:
    with open(repo+'p1d_data/data_files/MP-Gadget_data/1024_mock_%s.json' % str(sim_num), 'w') as f:
        json.dump(saveDict, f)
