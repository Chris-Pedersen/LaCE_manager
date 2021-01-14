import os
import sys
import json
import configargparse
from shutil import copy
import read_gadget
import write_submit_skewers_dirac as wsd

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--simdir', type=str, help='Base directory to simulation (crashes if it does not exist)', required=True)
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
simdir=args.simdir

wsd.write_skewer_scripts_in_sim(simdir=simdir,
                n_skewers=args.n_skewers,width_Mpc=args.width_Mpc,
                scales_T0=args.scales_T0,scales_gamma=args.scales_gamma,
                time=args.time,zmax=args.zmax,
                verbose=verbose,run=args.run)

