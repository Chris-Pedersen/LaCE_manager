import data_PD2013
import numpy as np
import json
import matplotlib.pyplot as plt
import os

'''
Script to overplot MP-Gadget simulated P1D(k)
with Nathalie's 2013 P1D data for a range of redshifts.
'''

## Load in MP-Gadget data
assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
repo=os.environ['LYA_EMU_REPO']
sim_data_path=repo+"/p1d_data/data_files/MP-Gadget_data/1024_L90_mimic.json"
with open(sim_data_path) as json_file:
    sim_data = json.load(json_file)

sim_z=[]
for item in sim_data:
    sim_z.append(item["z"])

## Load in PD_2013 data
realData=data_PD2013.P1D_PD2013(blind_data=False)
realData_z=realData.z

cm=plt.get_cmap('gnuplot')

max_z=max(max(sim_z),max(realData_z))
min_z=min(min(sim_z),min(realData_z))

data_k=realData.k

def rescale_zs(z):
    ### Function to rescale the redshifts
    ### to between 0 and 1 for the colormap
    return (z-min_z)/(max_z-min_z)

print(realData_z)
print("ks")
print(max(data_k))
plt.figure()
plt.title("P1Ds in the range 5>z>2 for PD13 (solid) and MP-Gadget (dashed)")
## Plot real data
for iz in range(len(realData_z)):
    p1d=realData.get_Pk_iz(iz)
    plt.semilogy(data_k,data_k*p1d,color=cm(rescale_zs(realData_z[iz])))

## Plot simulated data
for item in sim_data:
    p1d=np.asarray(item["p1d_kms"])
    k=np.asarray(item["k_kms"])
    z=item["z"]
    plt.semilogy(k,p1d*k,color=cm(rescale_zs(z)),linestyle="dashed")

plt.ylabel("k*p1d(k)")
plt.xlabel("k s/km")
plt.xlim(min(data_k)-0.001,max(data_k)+0.001)
plt.show()
