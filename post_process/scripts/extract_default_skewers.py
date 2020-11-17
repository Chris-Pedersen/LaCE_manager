""" Extract skewers for a given snapshot, using default temperature. """

import numpy as np
import argparse
# our modules below
import extract_skewers
import read_gadget
import json

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--simdir', type=str, help='Base simulation directory',required=True)
parser.add_argument('--skewers_dir', type=str, help='Store skewers in this folder',required=True)
parser.add_argument('--snap_num', type=int, default=8, help='Snapshot number',required=False)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")

# extract skewers for one snapshot, and return some info 
info=extract_skewers.rescale_write_skewers_z(simdir=args.simdir,
            num=args.snap_num, skewers_dir=args.skewers_dir,
            n_skewers=args.n_skewers,width_Mpc=args.width_Mpc)

if args.verbose:
    print('print extra info about skewers')
    print(info)

print('DONE')
