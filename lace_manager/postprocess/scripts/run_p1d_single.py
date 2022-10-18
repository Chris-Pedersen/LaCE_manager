import configargparse
from lace_manager.postprocess import write_p1d_script as wps

"""
For a single sim, read the fake_spectra files,
calculate the p1d and write an archive-format .json file to
store the mock p1d and parameters for a given training point.
"""

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--post_pair_dir', type=str,
                help='Base directory with simulation pair post-processings',required=True)
parser.add_argument('--n_skewers', type=int, default=10, help='Number of skewers per side',required=False)
parser.add_argument('--width_Mpc', type=float, default=0.1, help='Cell width (in Mpc)',required=False)
parser.add_argument('--scales_tau', type=str, default='1.0', help='Comma-separated list of optical depth scalings to use.',required=False)
parser.add_argument('--time', type=str, default='01:00:00', help='String formatted time to pass to SLURM script')
parser.add_argument('--zmax', type=float, default=5.5, help='Measure p1d for snapshots below this redshift')
parser.add_argument('--p1d_label', type=str, default=None, help='String identifying P1D measurement and / or tau scaling.',required=False)
parser.add_argument('--add_p3d', action='store_true', help='Measure also 3D P(k)')
parser.add_argument('--run', action='store_true', help='Actually submit the SLURM scripts')
parser.add_argument('--machine', type=str, default="hypatia", help='Specify machine where scripts are run (hypatia, cori)',required=False)
parser.add_argument('--queue', type=str, default="debug", help='Specify queue to use at NERSC (debug, regular)',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)

args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values())
print("----------")

for sim in ['sim_plus','sim_minus']:
    # label identifying this sim (to be used in full path)
    post_sim_dir='{}/{}/'.format(args.post_pair_dir,sim)
    wps.write_p1d_scripts_in_sim(post_dir=post_sim_dir,
            n_skewers=args.n_skewers,width_Mpc=args.width_Mpc,
            scales_tau=args.scales_tau,add_p3d=args.add_p3d,
            time=args.time,zmax=args.zmax,
            verbose=args.verbose,p1d_label=args.p1d_label,
            run=args.run,machine=args.machine,
            queue=args.queue)
