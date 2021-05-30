import os
import sys
import json
import configargparse
from lace.setup_simulations import read_gadget
from lace.postprocess import write_genpk_script as wgs

"""
Run Keir's modified GenPK on all sims in a specified
LH suite
"""

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--raw_dir', type=str,
                help='Base directory with raw simulation outputs (crashes if it does not exist)',required=True)
parser.add_argument('--post_dir', type=str,
                help='Base directory with simulation post-processings',required=True)
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
raw_dir=args.raw_dir
post_dir=args.post_dir

# read information about the hypercube
cube_json=raw_dir+'/latin_hypercube.json'
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
    if verbose:
        print('writing scripts for sample point',sample)
    for sim in ['sim_plus','sim_minus']:
        # label identifying this sim (to be used in full path)
        sim_tag='/sim_pair_{}/{}/'.format(sample,sim)
        wgs.write_genpk_scripts_in_sim(raw_dir=raw_dir+sim_tag,
                    post_dir=post_dir+sim_tag,
                    time=args.time,verbose=verbose)

