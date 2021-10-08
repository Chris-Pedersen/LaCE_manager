import matplotlib ## Suppresses plotting issues on compute nodes
matplotlib.use("Agg")
import shutil
import os
from lace.sampler import emcee_sampler

""" Script to resume an emcee chain from a backend.
Can run either until a given number of steps, of for a set
amount of time """

subfolder=None
rootdir=None

sampler=emcee_sampler.EmceeSampler(read_chain_file=13,
                    rootdir=rootdir,subfolder=subfolder)

## Cannot call self.log_prob using multiprocess.pool
def log_prob(theta):
    return sampler.like.log_prob_and_blobs(theta)

sampler.resume_sampler(1000,log_prob)

## Have to run this to write a new config.json
## with the updated chains.
sampler.write_chain_to_file()

## Copy corresponding job files to save folder
jobstring=jobstring="job"+os.environ['SLURM_JOBID']+".out"
slurmstring="slurm-"+os.environ['SLURM_JOBID']+".out"
shutil.copy(jobstring,sampler.save_directory+"/"+jobstring)
shutil.copy(slurmstring,sampler.save_directory+"/"+slurmstring)
