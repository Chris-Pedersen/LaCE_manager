import numpy as np
import configargparse
import os
import json
# our modules below
from lace_manager.postprocess import snapshot_admin
from lace_manager.postprocess import extract_skewers
from lace_manager.postprocess import measure_flux_power
from lace_manager.setup_simulations import read_gadget

"""
Script will run the final step of the postprocessing, the calculation of the
p1d from the spectra files, done in series. This part is relatively fast
and doesn't need to be done across multiple jobs in parallel. Also saves
us spamming the queue with thousands of jobs
"""

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--raw_dir', type=str,
                help='Base directory with raw simulation outputs (crashes if it does not exist)',required=True)
parser.add_argument('--post_dir', type=str,
                help='Base directory with simulation post-processings',required=True)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--scales_tau', type=str, default='[1.0]', help='Comma-separated list of optical depth scalings to use.',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--p1d_label', type=str, default=None, help='String identifying P1D measurement and / or tau scaling.',required=False)
args = parser.parse_args()

print(args.scales_tau)

scales_tau=[float(scale) for scale in args.scales_tau[1:-1].split(',')]

print('will scale tau by',scales_tau)

# read information about the hypercube
cube_json=args.raw_dir+"/latin_hypercube.json"
if not os.path.isfile(cube_json):
    raise ValueError('could not find hypercube '+cube_json)

with open(cube_json) as json_data:
    cube_data = json.load(json_data)

# get number of samples in the hyper-cube
nsamples=cube_data['nsamples']

# the code below is copy-pasted from single_sim_scripts/archive_flux_power.py

# for each sample, extract skewers for each snapshot
for sample in range(nsamples):
    for sim in ['sim_plus','sim_minus']:
        # label identifying this sim (to be used in full path)
        sim_tag='/sim_pair_{}/{}/'.format(sample,sim)
        skewers_dir=args.post_dir+sim_tag+'/skewers/'

        # get redshifts / snapshots Gadget parameter file 
        paramfile=args.raw_dir+sim_tag+'/paramfile.gadget'
        zs=read_gadget.redshifts_from_paramfile(paramfile)
        Nsnap=len(zs)

        ## Loop over snapshots
        for snap in range(Nsnap):
            # try to read information about filtering length in simulation
            kF_json=args.post_dir+sim_tag+'/filtering_length.json'
            if os.path.isfile(kF_json):
                # read json file with filtering data
                with open(kF_json) as json_data:
                    kF_data = json.load(json_data)
                kF_Mpc=kF_data['kF_Mpc'][snap]
                print('read kF_Mpc =',kF_Mpc)
            else:
                kF_Mpc=None

            # read file with information on temperature rescalings in snapshot
            fname=extract_skewers.get_snapshot_json_filename(
                                num=snap,n_skewers=args.n_skewers,
                                width_Mpc=args.width_Mpc)
            snap_filename=skewers_dir+'/'+fname
            
            # create an object that will deal with all skewers in the snapshot
            snapshot=snapshot_admin.SnapshotAdmin(snap_filename,
                                scales_tau=scales_tau,
                                kF_Mpc=kF_Mpc)
            # measure power for all tau scalings, for all temperature scalings
            archive_p1d=snapshot.get_all_flux_power()
            # write all measured power in a JSON file
            snapshot.write_p1d_json(p1d_label=args.p1d_label)

print('DONE')
