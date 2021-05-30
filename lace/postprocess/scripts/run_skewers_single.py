from lace.setup_simulations import read_gadget
from lace.postprocess import write_skewers_scripts as wss

"""
Script to run fake_spectra on all snapshots for a given simulation
"""

scales_T0='1.0'
scales_gamma='1.0'
n_skewers=500
width_Mpc=0.05
time='24:00:00'
run=True
zmax=6

# directory with raw simulation outputs
raw_dir='/data/desi/common/HydroData/Emulator/sims_256/sim_pair_1/'
#raw_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# directory with simulation post-procesings
post_dir='/data/desi/common/HydroData/Emulator/test_256/sim_pair_1/'
#post_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

for sim in ['sim_plus','sim_minus']:
    wss.write_skewer_scripts_in_sim(raw_dir+'/'+sim,post_dir+'/'+sim,
            n_skewers=n_skewers,width_Mpc=width_Mpc,
            scales_T0=scales_T0,scales_gamma=scales_gamma,
            time=time,zmax=zmax,
            verbose=True,run=run)

