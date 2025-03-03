#!/bin/bash -l
#SBATCH -J promptumamba_train
#SBATCH --time=3-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=v100l:4
#SBATCH --cpus-per-task=6          
#SBATCH --mem=96GB                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

# Set PROJECT_ROOT and PYTHONPATH
export PROJECT_ROOT=/home/nicocarp/scratch/PromptUMamba
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


config=$PROJECT_ROOT/configs/train_config.yaml

module purge

# Load necessary modules
module load StdEnv/2023
module load gcc/12.3
module load hdf5/1.14.2
module load cuda/12.2
module load cudnn/8.9.5.29
module load nccl/2.18.3
module load python/3.10

# Create and activate a virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip

# Install packages from Compute Canada wheels
pip install --no-index -r $PROJECT_ROOT/configs/env_local.txt

# Install packages from PyPI or other sources
pip install -r $PROJECT_ROOT/configs/env_pypi.txt

srun python main.py fit \
    --config configs/base.yaml \
    --config configs/model/pmr-plus.yaml \
    --config configs/train/pmr-plus/cmr24-cardiac.yaml 