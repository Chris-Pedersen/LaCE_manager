import numpy as np
import os
import read_gadget
import flux_real_genpk

def get_submit_string(options,time,output_files):
    submit_string='''#!/bin/bash
#!
#! Example SLURM job script for Peta4-Skylake (Skylake CPUs, OPA)
#! Last updated: Mon 13 Nov 12:25:17 GMT 2017
#! sbatch directives begin here ###############################
#SBATCH -J run_genpk
#SBATCH -A dirac-dp132-cpu
#SBATCH -o %s.out
#SBATCH -e %s.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=%s
#SBATCH -p skylake-himem

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\\1/')
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment
module load python-3.6.1-gcc-5.4.0-23fr5u4
module load gsl/2.4

#! Insert additional module load commands after this line if needed:
export PYTHONPATH=$HOME/lya_sims/lib/python3.6/

#! Full path to application executable: 
lya_scripts="/home/dc-font1/Codes/LyaCosmoParams/post_process/scripts"
application="python $lya_scripts/run_genpk_flux_real.py"

# setup options 
options="%s"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 32:
export OMP_NUM_THREADS=32

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets

CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
'''%(output_files,output_files,time,options)
    return submit_string
    

def get_options_string(simdir,snap_num,verbose):
    """ Option string to pass to python script in SLURM"""

    options='--simdir {} --snap_num {} '.format(simdir,snap_num)
    if verbose:
        options+='--verbose'

    return options


def write_genpk_script(script_name,simdir,snap_num,time,verbose):
    """ Generate a SLURM file to run GenPk for a given snapshot."""

    # construct string with options to be passed to python script
    options=get_options_string(simdir,snap_num,verbose)

    if verbose:
        print('print options: '+options)

    # set output files (.out and .err)
    output_files=simdir+'/slurm_genpk_'+str(snap_num)

    # get string with submission script
    submit_string=get_submit_string(options,time,output_files)

    submit_script = open(script_name,'w')
    for line in submit_string:
        submit_script.write(line)
    submit_script.close()


def write_genpk_scripts_in_sim(simdir,time,verbose):
    """ Generate a SLURM file for each snapshot to run GenPk"""
    
    if verbose:
        print('in write_genpk_scripts_in_sim',simdir)

    # get redshifts / snapshots Gadget parameter file 
    paramfile=simdir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    for snap in range(Nsnap):
        # figure out if GenPk was already computed
        genpk_filename=flux_real_genpk.flux_real_genpk_filename(simdir,snap)
        print('genpk filename =',genpk_filename)
        if os.path.exists(genpk_filename):
            if verbose: print('GenPk file existing',genpk_filename)
            continue
        else:
            if verbose: print('Will generate genpk file',genpk_filename)

        slurm_script=simdir+'/genpk_%s.sub'%snap
        write_genpk_script(script_name=slurm_script,simdir=simdir,
                            snap_num=snap,time=time,verbose=verbose)
        info_file=simdir+'/info_sub_genpk_'+str(snap)
        if verbose:
            print('print submit info to',info_file)
        cmd='sbatch '+slurm_script+' > '+info_file
        os.system(cmd)

