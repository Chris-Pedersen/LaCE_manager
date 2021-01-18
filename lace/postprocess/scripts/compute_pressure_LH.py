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
parser.add_argument('--basedir', type=str, help='Base directory to simulation suite (crashes if it does not exist)', required=True)
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
basedir=args.basedir
if args.kmax_Mpc > 0.0: 
    kmax_Mpc = args.kmax_Mpc
else:
    kmax_Mpc = None

# read information about the hypercube (to know number of simulations to use)
cube_json=basedir+'/latin_hypercube.json'
if not os.path.isfile(cube_json):
    print('could not find hypercube '+cube_json)
    print('assuming you passed a single simulation to analyze')
    filtering_length.fit_filtering_length(basedir,kmax_Mpc=kmax_Mpc,
                        write_json=True,show_plots=show_plots,verbose=verbose)
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
    # full path to folder for this particular simulation pair
    pair_dir=basedir+'/sim_pair_'+str(sample)
    if verbose:
        print('compute pressure history pair in',pair_dir)

    for sim in ['sim_plus','sim_minus']:
        simdir=pair_dir+'/'+sim
        filtering_length.fit_filtering_length(simdir,kmax_Mpc=kmax_Mpc,
                        write_json=True,show_plots=show_plots,verbose=verbose)


