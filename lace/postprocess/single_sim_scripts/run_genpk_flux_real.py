""" Run GenPk to measure power sepctrum of real flux."""

import argparse
from lace.postprocess import flux_real_genpk

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--raw_dir', type=str,
        help='Base directory with raw simulation outputs',required=True)
parser.add_argument('--post_dir', type=str,
        help='Base directory with simulation post-processings',required=True)
parser.add_argument('--snap_num', type=int, help='Snapshop number',required=True)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

flux_real_genpk.compute_flux_real_power(raw_dir=args.raw_dir,
        post_dir=args.post_dir,snap_num=args.snap_num,verbose=args.verbose)

