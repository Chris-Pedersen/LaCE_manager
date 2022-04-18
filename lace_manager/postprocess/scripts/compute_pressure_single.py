from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import filtering_length

""" This step of the postprocessing does not involve job submission scripts
as the fits are computationally fast """

verbose=True
show_plots=False

pair_dir="/share/rcifdata/chrisp/Aus20_Kathleens/P18"

kmax_Mpc = 15

for sim in ['sim_plus','sim_minus']:
    simdir=pair_dir+'/'+sim
    filtering_length.fit_filtering_length(simdir,kmax_Mpc=kmax_Mpc,
                    write_json=True,show_plots=show_plots,verbose=verbose)
