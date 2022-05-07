from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import filtering_length

""" This step of the postprocessing does not involve job submission scripts
as the fits are computationally fast """

verbose=True
show_plots=False

# directory with raw simulation outputs
raw_dir='/data/desi/common/HydroData/Emulator/sims_256/sim_pair_0/'
#raw_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# directory with simulation post-procesings
post_dir='/data/desi/common/HydroData/Emulator/test_256/sim_pair_0'
#post_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

kmax_Mpc = 15

for sim in ['sim_plus','sim_minus']:
    filtering_length.fit_filtering_length(raw_dir+'/'+sim,post_dir+'/'+sim,
                        kmax_Mpc=kmax_Mpc,write_json=True,
                        show_plots=show_plots,verbose=verbose)
