import matplotlib
matplotlib.use("Agg")
import numpy as np
import argparse
import json
# our modules below
from lace.postprocess import extract_skewers
from lace.setup_simulations import read_gadget

""" Extract skewers for a given snapshot, using different temperatures. """


# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--simdir', type=str, help='Base simulation directory',required=True)
parser.add_argument('--snap_num', type=int, help='Snapshop number',required=True)
parser.add_argument('--skewers_dir', type=str, help='Store skewers in this folder',required=False)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--scales_T0', type=str, default='1.0', help='Comma-separated list of T0 scalings to use.',required=False)
parser.add_argument('--scales_gamma', type=str, default='1.0', help='Comma-separated list of gamma scalings to use.',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

# main simulation folder 
simdir=args.simdir
if args.skewers_dir:
    skewers_dir=args.skewers_dir
else:
    skewers_dir=None

scales_T0=[float(scale) for scale in args.scales_T0.split(',')]
scales_gamma=[float(scale) for scale in args.scales_gamma.split(',')]

# should also read scales_T0 and scales_gamma
info=extract_skewers.rescale_write_skewers_z(simdir,num=args.snap_num,
            skewers_dir=skewers_dir,n_skewers=args.n_skewers,
            width_Mpc=args.width_Mpc,
            scales_T0=scales_T0,scales_gamma=scales_gamma)

print('DONE')
