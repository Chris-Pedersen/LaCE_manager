from lace.setup_simulations import read_gadget
from lace.postprocess import write_genpk_script as wgs

"""
Run Keir's modified GenPK on all snapshots of a single simulation
"""

pair_dir="/share/rcifdata/chrisp/Aus20_Kathleens/P18"
time="20:00:00"

for sim in ['sim_plus','sim_minus']:
    wgs.write_genpk_scripts_in_sim(simdir=pair_dir+'/'+sim,
                                    time=time,verbose=True)
