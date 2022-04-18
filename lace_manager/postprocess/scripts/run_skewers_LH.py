import os
import sys
import json
import configargparse
from lace.setup_simulations import read_gadget
from lace.postprocess import write_skewers_scripts as wss

"""
Script to run fake_spectra on all sims a specified LH suite
"""

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--basedir', type=str, help='Base directory to simulation suite (crashes if it does not exist)', required=True)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--scales_T0', type=str, default='1.0', help='Comma-separated list of T0 scalings to use.',required=False)
parser.add_argument('--scales_gamma', type=str, default='1.0', help='Comma-separated list of gamma scalings to use.',required=False)
parser.add_argument('--time', type=str, default='01:00:00', help='String formatted time to pass to SLURM script')
parser.add_argument('--zmax', type=float, default=5.5, help='Extract skewers for snapshots below this redshift')
parser.add_argument('--run', action='store_true', help='Actually submit the SLURM scripts')
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
basedir=args.basedir

# read information about the hypercube
cube_json=basedir+'/latin_hypercube.json'
if not os.path.isfile(cube_json):
    raise ValueError('could not find hypercube '+cube_json)

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
        print('writing scripts for pair in',pair_dir)

    for sim in ['sim_plus','sim_minus']:
        wss.write_skewer_scripts_in_sim(simdir=pair_dir+'/'+sim,
                n_skewers=args.n_skewers,width_Mpc=args.width_Mpc,
                scales_T0=args.scales_T0,scales_gamma=args.scales_gamma,
                time=args.time,zmax=args.zmax,
                verbose=verbose,run=args.run)

