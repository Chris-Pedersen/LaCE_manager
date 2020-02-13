import gp_emulator
import time

basedir='/home/chris/Projects/LyaCosmoParams/p1d_emulator/sim_suites/emulator_256_15072019'
p1d_label=None
skewers_label='Ns256_wM0.05'
undersample_z=1
paramList=["Delta2_p","mF","sigT_Mpc","gamma","kF_Mpc"]
max_arxiv_size=None
kmax_Mpc=8.0

start = time.time()
emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,kmax_Mpc=kmax_Mpc,
                               undersample_z=undersample_z,max_arxiv_size=max_arxiv_size,
                               verbose=False,paramList=paramList,train=True,asymmetric_kernel=True)
end=time.time()
print(end - start)
emu.saveEmulator()