""" Read skewers, rescale optical depth and measure 1D flux power. """

import numpy as np
import argparse
import os
import json
# our modules below
from lace.postprocess import snapshot_admin
from lace.postprocess import extract_skewers
from lace.postprocess import measure_flux_power
from lace.setup_simulations import read_gadget

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--post_dir', type=str,
                help='Base directory with simulation post-processings',required=True)
parser.add_argument('--snap_num', type=int, help='Snapshop number',required=True)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--scales_tau', type=str, default='1.0', help='Comma-separated list of optical depth scalings to use.',required=False)
parser.add_argument('--p1d_label', type=str, default=None, help='String identifying P1D measurement and / or tau scaling.',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

verbose=args.verbose
print('verbose =',verbose)

post_dir=args.post_dir

scales_tau=[float(scale) for scale in args.scales_tau.split(',')]
if verbose:
    print('will scale tau by',scales_tau)

# try to read information about filtering length in simulation
kF_json=post_dir+'/filtering_length.json'
if os.path.isfile(kF_json):
    # read json file with filtering data
    with open(kF_json) as json_data:
        kF_data = json.load(json_data)
    kF_Mpc=kF_data['kF_Mpc'][args.snap_num]
    print('read kF_Mpc =',kF_Mpc)
else:
    kF_Mpc=None

# read file containing information of all temperature rescalings in snapshot
snap_filename=post_dir+'/skewers/'+extract_skewers.get_snapshot_json_filename(
                num=args.snap_num,n_skewers=args.n_skewers,
                width_Mpc=args.width_Mpc)
if verbose:
    print('setup snapshot admin from file',snap_filename)

# create an object that will deal with all skewers in the snapshot
snapshot=snapshot_admin.SnapshotAdmin(snap_filename,scales_tau=scales_tau,
                                                            kF_Mpc=kF_Mpc)
Nsk=len(snapshot.data['sk_files'])
if verbose:
    print('snapshot has {} temperature rescalings'.format(Nsk))

# measure flux power for all tau scalings, for all temperature scalings
archive_p1d=snapshot.get_all_flux_power()

# write all measured power in a JSON file
snapshot.write_p1d_json(p1d_label=args.p1d_label)

print('DONE')
