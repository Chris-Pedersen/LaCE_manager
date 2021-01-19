from lace.setup_simulations import read_gadget
from lace.postprocess import write_p1d_script as wps

"""
For a single sim, read the fake_spectra files,
calculate the p1d and write an archive-format .json file to
store the mock p1d and parameters for a given training point.
"""

pair_dir="/share/rcifdata/chrisp/emulator_768_09122019/sim_pair_29"
n_skewers = 500
width_Mpc = 0.05
zmax = 6.0
scales_tau = '1.0'
time = '10:00:00'
run = True
p1d_label="p1d"
verbose=True

for sim in ['sim_plus','sim_minus']:
    wps.write_p1d_scripts_in_sim(simdir=pair_dir+'/'+sim,
            n_skewers=n_skewers,width_Mpc=width_Mpc,
            scales_tau=scales_tau,
            time=time,zmax=zmax,
            verbose=verbose,p1d_label=p1d_label,run=run)
