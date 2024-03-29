#!/bin/bash -l

#SBATCH -C haswell
#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=test_sampler
#SBATCH --output=/global/cfs/cdirs/desi/users/font/LaCE_manager/lace_manager/sampler/logs/test_sampler-%j.out
#SBATCH --error=/global/cfs/cdirs/desi/users/font/LaCE_manager/lace_manager/sampler/logs/test_sampler-%j.err

# load modules to use LaCE
module load python
module load gsl
source activate lace_env

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

sampler_dir="/global/cfs/cdirs/desi/users/font/LaCE_manager/lace_manager/sampler/"
echo "sampler dir", $sampler_dir

python -u $sampler_dir/scripts/multiprocess_sampler.py -c $sampler_dir/scripts/small_test.config --timeout 0.4 --burn_in 100 --prior_Gauss_rms 0.05 --nersc

