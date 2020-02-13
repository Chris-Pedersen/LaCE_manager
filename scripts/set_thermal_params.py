import numpy as np
import copy as cp
import astropy.units as u
import matplotlib.pyplot as plt
import os
import thermal_evolution as te


repo=os.environ['LYA_EMU_REPO']
data_filename = repo+'/setup_simulations/IGM_data/256suite_IGM_tables.txt'

thermal_evolution_instance = te.MPGadgetThermalEvolution(data_filename)
thermal_evolution_instance.train_interpolator(2., use_parameter=[False, True, True, True])

T0_low=9000
T0_high=15000

kF_low=8
kF_high=13

gamma_low=1.3
gamma_high=1.8

T0_samples=np.random.uniform(T0_low,T0_high,size=4)
gamma_samples=np.random.uniform(gamma_low,gamma_high,size=4)
kF_samples=np.random.uniform(kF_low,kF_high,size=4)

## Convert kF_samples into lambda_p (ckpc)
lambda_p_samples=1000*(1/kF_samples)

print(lambda_p_samples)

out=[]
for aa in range(len(T0_samples)):
    A=thermal_evolution_instance.predict_A([T0_samples[aa],
                                gamma_samples[aa],
                                lambda_p_samples[aa]])
    B=thermal_evolution_instance.predict_B([T0_samples[aa],
                                gamma_samples[aa],
                                lambda_p_samples[aa]])  
    z_rei=thermal_evolution_instance.predict_z_reionisation([T0_samples[aa],
                                gamma_samples[aa],
                                lambda_p_samples[aa]])
    samples=np.array([T0_samples[aa],
                        gamma_samples[aa],
                        kF_samples[aa],
                        A,B,z_rei])
    out.append(samples)


np.savetxt("samples.txt",out,delimiter=' ',header="T0, gamma, kF, A, B, z_rei")


