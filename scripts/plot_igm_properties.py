import p1d_arxiv
import numpy as np
import matplotlib.pyplot as plt
import mean_flux_model
from scipy.optimize import curve_fit
import os

repo=os.environ['LYA_EMU_REPO']
basedir=repo+"/p1d_emulator/sim_suites/emulator_512_18062019"


MF=mean_flux_model.MeanFluxModel()
archive=p1d_arxiv.ArxivP1D(basedir=basedir,pick_sim_number=2,
                            drop_tau_rescalings=True,
                            drop_temp_rescalings=True)

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

## x data is log10(1+z/(1+z_*)) (why????)

def get_mf(z,a,b):
    z_log=np.log((1+z)/(1+3))
    ln_tau_poly=np.poly1d([a,b])
    ln_tau=ln_tau_poly(z_log)
    return np.exp(-np.exp(ln_tau))

def get_power_law(z,a,b,c):
    z=np.log(z)
    log_poly=np.poly1d([a,b,c])
    ln_f=log_poly(z)
    return np.exp(ln_f)

def get_power_1storder(z,a,b):
    z=np.log(z)
    log_poly=np.poly1d([a,b])
    ln_f=log_poly(z)
    return np.exp(ln_f)

def get_broken_power(z,a,b,c):
    pivot=3.6
    out=np.empty(len(z))
    for aa in range(len(z)):
        lnz=np.log(z[aa]/pivot)
        if z[aa]<pivot:
            log_poly=np.poly1d([a,b])
            ln_f=log_poly(lnz)
            out[aa]=ln_f
        else:
            log_poly=np.poly1d([c,b])
            ln_f=log_poly(lnz)
            out[aa]=ln_f
    return out

    

z_fits=np.linspace(5,2,200)


opt_mf, cov_mf = curve_fit(get_mf, data[:,0], data[:,1])
power_gamma, cov_gamma=curve_fit(get_power_law,data[:,0],data[:,3])
power_kf, cov_kf=curve_fit(get_power_1storder,data[:,0],data[:,4])
broken_power_t0,broken_power_cov=curve_fit(get_broken_power,data[:,0],data[:,2])

mean_fluxes=get_mf(z_fits,opt_mf[0],opt_mf[1])
gamma_power=get_power_law(z_fits,power_gamma[0],power_gamma[1],power_gamma[2])
kf_power=get_power_1storder(z_fits,power_kf[0],power_kf[1])

broken_t0=get_broken_power(z_fits,broken_power_t0[0],
                                    broken_power_t0[1],
                                    broken_power_t0[2])


plt.figure(figsize=(8,15))
plt.subplot(4,1,1)
plt.plot(data[:,0],data[:,1],label="Sim")
plt.plot(z_fits,mean_fluxes,label="Model")
plt.ylabel("<F>")
plt.legend()
plt.xticks([])

plt.subplot(4,1,2)
plt.plot(data[:,0],data[:,2])
plt.plot(z_fits,broken_t0)
plt.ylabel("T0")
plt.xticks([])

plt.subplot(4,1,3)
plt.plot(data[:,0],data[:,3])
plt.plot(z_fits,gamma_power)
plt.ylabel("gamma")
plt.xticks([])

plt.subplot(4,1,4)
plt.plot(data[:,0],data[:,4])
plt.plot(z_fits,kf_power)
plt.ylabel("kF")
plt.tight_layout()
plt.show()

'''


zs=np.empty(len(archive.data))
T0_array=np.empty(len(zs))
gamma_array=np.empty(len(zs))
mean_fluxes=np.empty(len(zs))


for aa in range(len(archive.data)):
    zs[aa]=archive.data[aa]["z"]
    T0_array[aa]=archive.data[aa]["T0"]
    gamma_array[aa]=archive.data[aa]["gamma"]
    mean_fluxes[aa]=MF.get_mean_flux(zs[aa])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(zs, T0_array, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('gamma', color=color)  # we already handled the x-label with ax1
ax2.plot(zs, gamma_array, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.xlim(5,2)
plt.show()
'''
