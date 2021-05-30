from lace.setup_simulations import read_gadget
from lace.postprocess import write_genpk_script as wgs

"""
Run Keir's modified GenPK on all snapshots of a single simulation
"""

time="20:00:00"

# directory with raw simulation outputs
raw_dir='/data/desi/common/HydroData/Emulator/sims_256/'
#raw_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# directory with simulation post-procesings
post_dir='/data/desi/common/HydroData/Emulator/test_256/'
#post_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# label identifying simulation pair
pair_tag='sim_pair_0'

for sim in ['sim_plus','sim_minus']:
    sim_tag='/'+pair_tag+'/'+sim
    wgs.write_genpk_scripts_in_sim(raw_dir+sim_tag,post_dir+sim_tag,
                        time=time,verbose=True)
