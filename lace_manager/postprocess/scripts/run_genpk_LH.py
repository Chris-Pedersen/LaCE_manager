import os
import sys
import json
import configargparse
from shutil import copy
from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import write_genpk_script as wgs

"""
Run Keir's modified GenPK on all sims in a specified
LH suite
"""

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--basedir', type=str, help='Base directory to simulation suite (crashes if it does not exist)', required=True)
parser.add_argument('--time', type=str, default='00:30:00', help='String formatted time to pass to SLURM script')
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

# for each sample, run genpk in all snapshots
for sample in range(nsamples):
    # full path to folder for this particular simulation pair
    pair_dir=basedir+'/sim_pair_'+str(sample)
    if verbose:
        print('writing scripts for pair in',pair_dir)

    for sim in ['sim_plus','sim_minus']:
        wgs.write_genpk_scripts_in_sim(simdir=pair_dir+'/'+sim,
                                        time=args.time,verbose=verbose)

