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
import p1d_arxiv
import matplotlib.animation as animation

'''
Script to check how close our fiducial models for each mock data
sim actually reproduce the mock data
'''

z_emu=True ## Use a redshift - split emulator?
drop3ds=True
test_sim_number=145 ## Pick a sim between 0 and 199

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


archive=p1d_arxiv.ArxivP1D(basedir=basedir,
                            drop_tau_rescalings=True,z_max=4,
                            drop_sim_number=test_sim_number,
                            drop_temp_rescalings=True,skewers_label=skewers_label)
mock=p1d_arxiv.ArxivP1D(basedir=basedir,
                            drop_tau_rescalings=True,z_max=4,
                            pick_sim_number=test_sim_number,
                            drop_temp_rescalings=True,skewers_label=skewers_label)

if z_emu:
    z_list=[]
    for entry in mock.data:
        z_list.append(entry["z"])
    emu=z_emulator.ZEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",z_list=z_list,passArxiv=archive,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)
else:
    emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,z_max=4,
                                verbose=False,paramList=paramList,train=True,
                                emu_type="k_bin",passArxiv=archive,
                                drop_tau_rescalings=True,
                                drop_temp_rescalings=True)

test_k=np.logspace(np.log10(mock.data[0]["k_Mpc"][1]),np.log10(mock.data[0]["k_Mpc"][len(emu.training_k_bins)-2]),500)

#plt.figure()
for aa, zz in enumerate(mock.z):
    col = plt.cm.jet(aa/(len(mock.z)-1))
    pred,err=emu.emulate_p1d_Mpc(mock.data[aa],test_k,return_covar=True,z=zz)
    dist=emu.get_nearest_distance(mock.data[aa],z=zz)
    plt.plot(mock.data[aa]["k_Mpc"][1:],mock.data[aa]["p1d_Mpc"][1:]*mock.data[aa]["k_Mpc"][1:],
    color=col,label="z=%.3f, distance=%.4f" % (mock.z[aa],dist),marker="o")
    plt.plot(test_k,pred*test_k,color=col,linestyle="dashed")
    plt.fill_between(test_k,(pred+np.sqrt(np.diag(err)))*test_k,(pred-np.sqrt(np.diag(err)))*test_k,color=col,alpha=0.3)
plt.xlim(min(test_k),max(test_k))
plt.title("z emulator=%s, no <F> rescalings" % z_emu)
plt.xlabel(r"$k_\parallel$ [1/Mpc]")
plt.ylabel(r"$k_\parallel$*P1D($k_\parallel$)")
plt.legend()
plt.yscale("log")

if z_emu or drop3ds:
    plt.show()
    quit()

emu_1=[]
emu_2=[]
emu_3=[]
emu_4=[]
emu_5=[]

mock_1=[]
mock_2=[]
mock_3=[]
mock_4=[]
mock_5=[]

# figure out values of param_1,param_2 in arxiv
for model in emu.arxiv.data:
    emu_1.append(model["mF"])
    emu_2.append(model["sigT_Mpc"])
    emu_3.append(model["gamma"])
    emu_4.append(model["Delta2_p"])
    emu_5.append(model["kF_Mpc"])
for model in mock.data:
    mock_1.append(model["mF"])
    mock_2.append(model["sigT_Mpc"])
    mock_3.append(model["gamma"])
    mock_4.append(model["Delta2_p"])
    mock_5.append(model["kF_Mpc"])

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(emu_1, emu_2, emu_3,s=4,label="Training points",color="blue")
ax.scatter3D(mock_1,mock_2,mock_3,s=14,label="Test point",color="red")
ax.set_xlabel("mF")
ax.set_ylabel("sigT_Mpc")
ax.set_zlabel("gamma")

def rotate(angle):
    ax.view_init(azim=angle)

print("Making animation")
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 250, 1), interval=50)
rot_animation.save('rot1.gif', dpi=80, writer='imagemagick')

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(emu_3, emu_4, emu_5,s=4,label="Training points",color="blue")
ax.scatter3D(mock_3,mock_4,mock_5,s=14,label="Test point",color="red")
ax.set_xlabel("gamma")
ax.set_ylabel("delta^2")
ax.set_zlabel("kF_Mpc")


def rotate(angle):
    ax.view_init(azim=angle)

print("Making animation")
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 250, 1), interval=50)
rot_animation.save('rot2.gif', dpi=80, writer='imagemagick')

plt.show()
