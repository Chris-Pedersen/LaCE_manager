#!/bin/bash -l

#SBATCH -C haswell
#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=test_resume
#SBATCH --output=/global/cfs/cdirs/desi/users/font/LaCE/lace/sampler/logs/test_resume-%j.out
#SBATCH --error=/global/cfs/cdirs/desi/users/font/LaCE/lace/sampler/logs/test_resume-%j.err

# load modules to use LaCE
module load python
module load gsl
source activate lace_manager

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

sampler_dir="/global/cfs/cdirs/desi/users/font/LaCE/lace/sampler/"
echo "sampler dir", $sampler_dir

python -u $sampler_dir/scripts/resume_sampler.py --subfolder central_sim --chain_num 9 --timeout 0.4 --nersc
