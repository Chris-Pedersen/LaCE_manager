import numpy as np

def get_submit_string(simdir='blah',nodes=4,time='08:00:00'):
    submit_string='''#!/bin/bash
#! Example SLURM job script for Peta4-Skylake (Skylake CPUs, OPA)
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J MP-gadget
#SBATCH -A dirac-dp132-cpu
#SBATCH --nodes=%d
#SBATCH --ntasks=64
#SBATCH --time=%s 
#SBATCH --mail-type=NONE
#SBATCH -p skylake-himem

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\\1/')
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment
module load gsl/2.4

## Gen ICs
application="/home/dc-pede1/Codes/MP-Gadget-Stable/genic/MP-GenIC"
options=%s/paramfile.genic

workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}]

export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

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

## Now run sim
application="/home/dc-pede1/Codes/MP-Gadget-Stable/gadget/MP-Gadget"
options=%s/paramfile.gadget

workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}]

export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
'''%(nodes,time,simdir,simdir)
    return submit_string
    

def write_simulation_script(script_name='test.submit',simdir='test_script',
            nodes=2,time='01:00:00'):
    """ Generate a SLURM file to run both GenIC and MP-Gadget for a sim."""

    submit_string=get_submit_string(simdir,nodes,time)

    submit_script = open(script_name,'w')
    for line in submit_string:
        submit_script.write(line)
    submit_script.close()


