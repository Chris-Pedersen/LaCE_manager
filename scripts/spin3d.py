import p1d_arxiv
import numpy as np
import test_simulation
import gp_emulator
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.cm as cm

""" Script to plot the contributions from each training point
to the mean PPD prediction. Currently only uses the full kernel

NB there's no easy way to close the plots right now outside of just
killing the kernel. They will spin for 540 degrees (thanks Tony Hawk)
until closing """

test_sim_number=15
skewers_label='Ns500_wM0.05'
#skewers_label=None
#basedir="/p1d_emulator/sim_suites/emulator_256_28082019/"
basedir="/p1d_emulator/sim_suites/Australia20/"
#basedir=repo+"/p1d_emulator/sim_suites/emulator_256_15072019/"
p1d_label=None
undersample_z=12
paramList=['mF', 'sigT_Mpc', 'gamma', 'kF_Mpc', 'Delta2_p', 'n_p']
max_arxiv_size=None
kmax_Mpc=8
z_max=4.0
z_list=None
# load Latin hypercube in simulation space
archive=p1d_arxiv.ArxivP1D(basedir=basedir,
                            drop_tau_rescalings=True,
                            drop_temp_rescalings=True,skewers_label=skewers_label)


test_sim=test_simulation.TestSimulation(basedir,"central",skewers_label=skewers_label,z_max=5,
                kmax_Mpc=8,kp_Mpc=0.7)


## Build emulator
paramList=['mF', 'sigT_Mpc', 'gamma', 'kF_Mpc', 'Delta2_p', 'n_p']
# specify k range
kmax_Mpc=8
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,z_max=z_max,
                                verbose=False,paramList=paramList,train=False,
                                emu_type="k_bin",passArxiv=archive,
                                kmax_Mpc=kmax_Mpc)
emu.load_default()

emu_call_dict=test_sim.get_emulator_calls(3)
emu_call=emu.return_unit_call(emu_call_dict)

## Add this parameter vector to the X training data
emu_call=np.expand_dims(emu_call,axis=0) ## Expand number of dimensions to match X grid

## The following is essentially a list of parameter vectors
## The first entry is the test point defined above
## The remining entries are the position vectors for each training point
## Have to do it this way as GPy doesn't allow individual calculations of the
## covariances apparently
test_and_training=np.concatenate((emu_call,emu.X_param_grid),axis=0)


## Now calculate the covariance ##
## Linear only
C_lin=emu.gp.kern.linear.K(test_and_training)

## RBF only
C_rbf=emu.gp.kern.rbf.K(test_and_training)

## Full kernel
C_full=emu.gp.kern.K(test_and_training)

## Calculate W
K_inv_rbf=np.linalg.inv(C_rbf)
K_inv_lin=np.linalg.inv(C_lin)
K_inv_full=np.linalg.inv(C_full)

W_rbf=C_rbf[0]*K_inv_rbf
W_lin=C_lin[0]*K_inv_lin
W_full=C_full[0]*K_inv_full
####################################

## Select parameters along which to project
param1="n_p"
param2="Delta2_p"
param3="mF"
param4="sigT_Mpc"
param5="gamma"
param6="kF_Mpc"

emu_data=archive.data
Nemu=len(emu_data)


## Training points
emu_1=np.empty(Nemu)
emu_2=np.empty(Nemu)
emu_3=np.empty(Nemu)
emu_4=np.empty(Nemu)
emu_5=np.empty(Nemu)
emu_6=np.empty(Nemu)

ppd_weight_rbf=np.empty(Nemu)
ppd_weight_lin=np.empty(Nemu)
ppd_weight_full=np.empty(Nemu)

## Populate grids for plots
for aa in range(Nemu):
    emu_1[aa]=emu_data[aa][param1]
    emu_2[aa]=emu_data[aa][param2]
    emu_3[aa]=emu_data[aa][param3]
    emu_4[aa]=emu_data[aa][param4]
    emu_5[aa]=emu_data[aa][param5]
    emu_6[aa]=emu_data[aa][param6]
    ## Mean ppd contributions (W(X_star,X))
    ppd_weight_rbf[aa]=W_rbf[0][aa+1]
    ppd_weight_lin[aa]=W_lin[0][aa+1]
    ppd_weight_full[aa]=W_full[0][aa+1]

    
## Point where the emu calls were made
call_1=emu_call_dict[param1]
call_2=emu_call_dict[param2]
call_3=emu_call_dict[param3]
call_4=emu_call_dict[param4]
call_5=emu_call_dict[param5]
call_6=emu_call_dict[param6]

cmap=cm.PiYG
## Plot first set of params
zmax=max(ppd_weight_full)
zmin=-1.*zmax
fig = plt.figure(figsize=(11,8))
ax = plt.axes(projection="3d")
ax.scatter3D(call_1, call_2, call_3,c="red",s=25,vmin=zmin, vmax=zmax,marker="x")
p=ax.scatter3D(emu_1, emu_2, emu_3,c=ppd_weight_full,s=13,
            cmap=cmap,vmin=zmin, vmax=zmax,edgecolor="black",linewidth=0.3)
fig.colorbar(p)
ax.set_xlabel(param1)
ax.set_ylabel(param2)
ax.set_zlabel(param3)
# rotate the axes and update
for angle in range(0, 640):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
plt.close()

## Plot second set of params
zmax=max(ppd_weight_full)
zmin=-1.*zmax
fig = plt.figure(figsize=(11,8))
ax = plt.axes(projection="3d")
ax.scatter3D(call_4, call_5, call_6,c="red",s=25,vmin=zmin, vmax=zmax,marker="x")
p=ax.scatter3D(emu_4, emu_5, emu_6,c=ppd_weight_full,s=13,
            cmap=cmap,vmin=zmin, vmax=zmax,edgecolor="black",linewidth=0.3)
fig.colorbar(p)
ax.set_xlabel(param4)
ax.set_ylabel(param5)
ax.set_zlabel(param6)
# rotate the axes and update
for angle in range(0, 540):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
