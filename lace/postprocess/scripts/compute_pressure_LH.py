import os
import sys
import json
import configargparse
from lace.setup_simulations import read_gadget
from lace.postprocess import filtering_length

""" This step of the postprocessing does not involve job submission scripts
as the fits are computationally fast """

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--raw_dir', type=str,
        help='Base directory with raw simulation outputs (crashes if it does not exist)',required=True)
parser.add_argument('--post_dir', type=str,
        help='Base directory with simulation post-processings',required=True)
parser.add_argument('--kmax_Mpc', type=float,default=15.0, help='Fit pressure smoothing for k < kmax_Mpc', required=False)
parser.add_argument('--show_plots', action='store_true', help='Plot measured power and best fit filtering length',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)

args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values())
print("----------")

verbose=args.verbose
show_plots=args.show_plots
raw_dir=args.raw_dir
post_dir=args.post_dir
if args.kmax_Mpc > 0.0: 
    kmax_Mpc = args.kmax_Mpc
else:
    kmax_Mpc = None

# read information about the hypercube (to know number of simulations to use)
cube_json=raw_dir+'/latin_hypercube.json'
if not os.path.isfile(cube_json):
    print('could not find hypercube '+cube_json)
    print('assuming you passed a single simulation to analyze')
    filtering_length.fit_filtering_length(raw_dir=raw_dir,post_dir=post_dir,
                    kmax_Mpc=kmax_Mpc,write_json=True,
                    show_plots=show_plots,verbose=verbose)
    exit()

with open(cube_json) as json_data:
    cube_data = json.load(json_data)
if verbose:
    print('print cube info')
    print(cube_data)

# get number of samples in the hyper-cube
nsamples=cube_data['nsamples']

# for each sample, extract skewers for each snapshot
for sample in range(nsamples):
    if verbose:
        print('compute pressure for sample point',sample)
    for sim in ['sim_plus','sim_minus']:
        # label identifying this sim (to be used in full path)
        sim_tag='/sim_pair_{}/{}/'.format(sample,sim)
        filtering_length.fit_filtering_length(raw_dir=raw_dir+sim_tag,
                    post_dir=post_dir+sim_tag,kmax_Mpc=kmax_Mpc,
                    write_json=True,show_plots=show_plots,verbose=verbose)

