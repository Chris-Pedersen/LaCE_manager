""" Extract skewers for default values of temperature. """

import numpy as np
import argparse
# our modules below
import extract_skewers
import read_gadget
import json

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--simdir', type=str, help='Base simulation directory',required=True)
parser.add_argument('--skewers_dir', type=str, help='Store skewers in this folder',required=False)
parser.add_argument('--zmax', type=float, default=6.0, help='Extract skewers only for z < zmax',required=False)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_kms', type=float, default=10.0, help='Cell width (in km/s)',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

# main simulation folder 
simdir=args.simdir
if args.skewers_dir:
    skewers_dir=args.skewers_dir
else:
    skewers_dir=None

# extract skewers for all snapshots, and return info about mean flux
sk_info=extract_skewers.write_default_skewers(simdir=simdir,
            skewers_dir=skewers_dir,zmax=args.zmax,
            n_skewers=args.n_skewers,width_kms=args.width_kms)

if args.verbose:
    print('got mean flux information')
    print(sk_info)

# write skewers information in JSON file
filename=sk_info['skewers_dir']+'/default_skewers.json'
if args.verbose:
    print('will print skewer info to',filename)
json_file = open(filename,"w")
json.dump(sk_info,json_file)
json_file.close()

print('DONE')
