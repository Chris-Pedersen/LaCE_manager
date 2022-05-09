import os
import sys
import json
import configargparse
from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import write_p1d_script as wps

"""
For all sims in a LH suite, read the fake_spectra files,
calculate the p1d and write an archive-format .json file to
store the mock p1d and parameters for a given training point.
"""

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--post_dir', type=str,
                help='Base directory with simulation post-processings',required=True)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--scales_tau', type=str, default='1.0', help='Comma-separated list of optical depth scalings to use.',required=False)
parser.add_argument('--time', type=str, default='01:00:00', help='String formatted time to pass to SLURM script')
parser.add_argument('--zmax', type=float, default=5.5, help='Measure p1d for snapshots below this redshift')
parser.add_argument('--p1d_label', type=str, default=None, help='String identifying P1D measurement and / or tau scaling.',required=False)
parser.add_argument('--add_p3d', action='store_true', help='Measure also 3D P(k)')
parser.add_argument('--run', action='store_true', help='Actually submit the SLURM scripts')
parser.add_argument('--machine', type=str, default="hypatia", help='Specify machine where scripts are run (hypatia, cori)',required=False)
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
post_dir=args.post_dir

# read information about the hypercube
cube_json=post_dir+'/latin_hypercube.json'
if not os.path.isfile(cube_json):
    raise ValueError('could not find hypercube '+cube_json)
print(cube_json)

with open(cube_json) as json_data:
    cube_data = json.load(json_data)

if verbose:
    print('print cube info')
    print(cube_data)

# get number of samples in the hyper-cube
nsamples=cube_data['nsamples']

# for each sample, measure p1d for all skewers
for sample in range(nsamples):
    if verbose:
        print('writing scripts for sample point',sample)
    for sim in ['sim_plus','sim_minus']:
        # label identifying this sim (to be used in full path)
        sim_tag='/sim_pair_{}/{}/'.format(sample,sim)
        print('sim dir',post_dir+sim_tag)
        wps.write_p1d_scripts_in_sim(post_dir=post_dir+sim_tag,
                n_skewers=args.n_skewers,width_Mpc=args.width_Mpc,
                scales_tau=args.scales_tau,add_p3d=args.add_p3d,
                time=args.time,zmax=args.zmax,
                verbose=verbose,p1d_label=args.p1d_label,
                run=args.run,machine=args.machine)
