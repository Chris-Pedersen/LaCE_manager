import matplotlib ## Suppresses plotting issues on compute nodes
matplotlib.use("Agg")
import shutil
import os
import configargparse
import time
from lace_manager.sampler import emcee_sampler


""" Script to resume an emcee chain from a backend.
Can run either until a given number of steps, of for a set
amount of time """

os.environ["OMP_NUM_THREADS"] = "1"

parser = configargparse.ArgumentParser()
parser.add_argument('--subfolder', required=True, help='Mock dataset analyse (e.g., central_sim)')
parser.add_argument('--rootdir', default=None, help='Specify location of chains')
parser.add_argument('--chain_num', required=True, type=int, help='Number identifying chain to resume')
parser.add_argument('--nsteps', default=1000000, type=int, help='Stop chain after this number of (new) steps')
parser.add_argument('--timeout', default=47.5, type=float, help='Stop chain after these many hours')
parser.add_argument('--nersc', action='store_true', help='Running script at NERSC')
args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values())
print("----------")

# construct emcee sampler
sampler=emcee_sampler.EmceeSampler(read_chain_file=args.chain_num,
                    rootdir=args.rootdir,subfolder=args.subfolder)

## Cannot call self.log_prob using multiprocess.pool
def log_prob(theta):
    return sampler.like.log_prob_and_blobs(theta)

## specify number of steps
start = time.time()
sampler.like.go_silent()
sampler.resume_sampler(max_steps=args.nsteps,log_func=log_prob,
                    timeout=args.timeout)
end = time.time()
res_time = end - start
print("Sampling took {0:.1f} seconds".format(res_time))

## Have to run this to write a new config.json
## with the updated chains.
sampler.write_chain_to_file()

## below does not work at NERSC
if not args.nersc:
    ## Copy corresponding job files to save folder
    jobstring=jobstring="job"+os.environ['SLURM_JOBID']+".out"
    slurmstring="slurm-"+os.environ['SLURM_JOBID']+".out"
    shutil.copy(jobstring,sampler.save_directory+"/"+jobstring)
    shutil.copy(slurmstring,sampler.save_directory+"/"+slurmstring)
