import os
import sys
import json
import configargparse
from shutil import copy
sys.path.append('/home/dc-pede1/Codes/MP-Gadget-Master/tools/')
import write_submit_simulation_dirac as wsd

# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('--basedir', type=str, help='Base directory where all sims will be stored (crashes if it does not exist)',required=True)
parser.add_argument('--nodes', type=int, default=2, help='Number of nodes to use to run GenIC and MP-Gadget')
parser.add_argument('--time', type=str, default='01:00:00', help='String formatted time to pass to SLURM script')
parser.add_argument('--run', action='store_true', help='Actually submit the SLURM scripts (for now, only possible if total nodes < 100)')
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
basedir=args.basedir
nodes=args.nodes
time=args.time

# read information about the hypercube
cube_json=basedir+'/latin_hypercube.json'
with open(cube_json) as json_data:
    cube_data = json.load(json_data)
if verbose:
    print('print cube info')
    print(cube_data)

# get number of samples in the hyper-cube
nsamples=cube_data['nsamples']

# directory to LyaCosmoParams repo
lya_repo='/home/dc-pede1/Codes/LyaCosmoParams/'

# for each sample, run make_class_power and copy the files to the right path
for sample in range(nsamples):
    # full path to folder for this particular simulation pair
    pair_dir=basedir+'/sim_pair_'+str(sample)
    if verbose:
        print('writing scripts for pair in',pair_dir)

    # full path to one each simulation in the pair
    plus_dir=pair_dir+'/sim_plus/'
    minus_dir=pair_dir+'/sim_minus/'

    # write submission script to both simulations
    plus_submit=plus_dir+'/restart.sub'
    rsd.write_restart_simulation_script(script_name=plus_submit,
                    simdir=plus_dir,nodes=nodes,time=time)
    minus_submit=minus_dir+'/restart.sub'
    rsd.write_restart_simulation_script(script_name=minus_submit,
                    simdir=minus_dir,nodes=nodes,time=time)

    if args.run:
        total_nodes=2*args.nodes*nsamples
        if total_nodes < 1000:
            print('will submit scripts, for a total of {} nodes'.format(total_nodes))
            cmd_plus='sbatch '+plus_submit+' > '+plus_dir+'/info_sim_sub'
            os.system(cmd_plus)
            cmd_minus='sbatch '+minus_submit+' > '+minus_dir+'/info_sim_sub'
            os.system(cmd_minus)
        else:
            print('will NOT submit scripts, too many nodes = {}'.format(total_nodes))

