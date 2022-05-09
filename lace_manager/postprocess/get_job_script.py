import numpy as np
import os
from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import flux_real_genpk

def get_job_script(name,postprocess_script,options,time,output_files,machine="hypatia"):
    """ Return a job script
     - name: job name
     - postprocess_script: which postprocessing script to run
     - options: arguments to be passed to the executable
     - time: job time limit
     - output_files: name and path to save job log files
     - machine: specify machine (hypatia, cori) """

    if machine=="hypatia":
        return get_hypatia_script(name,postprocess_script,options,time,output_files)
    elif machine=="cori":
        print('ignore time in cori script')
        return get_cori_script(name,postprocess_script,options,output_files)
    else:
        raise ValueError("unknown machine "+machine)


def get_hypatia_script(name,postprocess_script,options,time,output_files):
    submit_string='''#!/bin/bash
#!
#! Example SLURM job script for Peta4-Skylake (Skylake CPUs, OPA)
#! Last updated: Mon 13 Nov 12:25:17 GMT 2017
#! sbatch directives begin here ###############################
#SBATCH -J %s
#SBATCH -o %s.out
#SBATCH -e %s.err
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=%s
#SBATCH -p CORES24
#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
## Load modules
module load python/3.6.4
module load hdf5/1.10.1
#! Full path to application executable:
lya_scripts="/home/chrisp/Codes/lace_manager/lace/postprocess/single_sim_scripts"
application="python3 $lya_scripts/%s"
# setup options 
options="%s"
#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.
export OMP_NUM_THREADS=24
np=$[${numnodes}*${mpi_tasks_per_node}]
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
CMD="$application $options"
cd $workdir
echo -e "Changed directory to `pwd`.
"
echo -e "JobID: $JOBID
======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "
Nodes allocated:
================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi
echo -e "
numtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "
Executing command:
==================
$CMD
"
eval $CMD'''%(name,output_files,output_files,time,postprocess_script,options)
    return submit_string


def get_cori_script(name,postprocess_script,options,output_files):
    submit_string='''#!/bin/bash
#SBATCH -C haswell
#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH -J %s
#SBATCH -o %s-%j.out
#SBATCH -e %s-%j.err

## Load modules
module load python
module load gsl
source activate lace_pp

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

#! Full path to application executable:
lya_scripts="/global/cfs/cdirs/desi/users/font/LaCE_pp/lace_manager/postprocess/single_sim_scripts"
application="python $lya_scripts/%s"

# setup options
options="%s"

CMD="$application $options"
echo -e "
Executing command:
==================
$CMD
"
eval $CMD'''%(name,output_files,output_files,postprocess_script,options)
    return submit_string


###################################################
## Scripts for other machines can be added below ##
###################################################
