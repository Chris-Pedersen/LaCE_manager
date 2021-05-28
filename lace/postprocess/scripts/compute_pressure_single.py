from lace.postprocess import filtering_length

""" This step of the postprocessing does not involve job submission scripts
as the fits are computationally fast """

verbose=True
show_plots=False

# directory with raw simulation outputs
raw_dir='/data/desi/common/HydroData/Emulator/sims_256/'
#raw_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# directory with simulation post-procesings
post_dir='/data/desi/common/HydroData/Emulator/test_256/'
#post_dir='/share/rcifdata/chrisp/Aus20_Kathleens/P18/'

# label identifying simulation pair
pair_tag='sim_pair_0'

kmax_Mpc = 15

for sim in ['sim_plus','sim_minus']:
    sim_tag='/'+pair_tag+'/'+sim
    filtering_length.fit_filtering_length(raw_dir+sim_tag,post_dir+sim_tag,
                        kmax_Mpc=kmax_Mpc,write_json=True,
                        show_plots=show_plots,verbose=verbose)
