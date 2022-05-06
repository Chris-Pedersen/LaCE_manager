#!/bin/bash -l

#SBATCH -C haswell
#SBATCH --partition=regular
#SBATCH --account=desi
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=sampler
#SBATCH --output=/global/cfs/cdirs/desi/users/font/LaCE/lace/sampler/logs/sampler-%j.out
#SBATCH --error=/global/cfs/cdirs/desi/users/font/LaCE/lace/sampler/logs/sampler-%j.err

# load modules to use LaCE
module load python
module load gsl
source activate lace_manager

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

sampler_dir="/global/cfs/cdirs/desi/users/font/LaCE/lace/sampler/"
echo "sampler dir", $sampler_dir

python -u $sampler_dir/scripts/multiprocess_sampler.py -c $sampler_dir/scripts/example_compressed.config --timeout 11.5 --nersc
