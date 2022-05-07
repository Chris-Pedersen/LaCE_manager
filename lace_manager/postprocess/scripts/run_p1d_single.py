from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import write_p1d_script as wps

"""
For a single sim, read the fake_spectra files,
calculate the p1d and write an archive-format .json file to
store the mock p1d and parameters for a given training point.
"""

n_skewers = 500
width_Mpc = 0.05
zmax = 6.0
scales_tau = '1.0'
time = '10:00:00'
run = True
p1d_label="p1d"
verbose=True

# directory with raw simulation outputs
raw_dir='/data/desi/common/HydroData/Emulator/sims_256/sim_pair_0/'
#raw_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# directory with simulation post-procesings
post_dir='/data/desi/common/HydroData/Emulator/test_256/sim_pair_0/'
#post_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

for sim in ['sim_plus','sim_minus']:
    wps.write_p1d_scripts_in_sim(raw_dir+'/'+sim,post_dir+'/'+sim,
            n_skewers=n_skewers,width_Mpc=width_Mpc,
            scales_tau=scales_tau,
            time=time,zmax=zmax,
            verbose=verbose,p1d_label=p1d_label,run=run)
