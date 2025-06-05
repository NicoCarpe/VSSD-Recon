#!/bin/bash -l
#SBATCH -J promptumamba_train
#SBATCH --time=00-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=v100l:4
#SBATCH --cpus-per-task=2       
#SBATCH --mem=64GB                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

module --force purge

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Point to project
export PROJECT_ROOT=/home/nicocarp/scratch/PromptUMamba
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# —————— WandB configuration ——————
# Load all variables from .env
set -o allexport
source $PROJECT_ROOT/.env
set +o allexport

# name project in WandB
export WANDB_PROJECT="PromptMamba_Training"

module load StdEnv/2023
module load gcc/12.3
module load hdf5/1.14.2
module load cuda/12.2
module load cudnn/8.9.5.29
module load nccl/2.18.3
module load python/3.10

# create a clean venv
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

python -m pip install --upgrade pip
python -m pip install --no-index -r $PROJECT_ROOT/configs/env_local.txt
python -m pip install -r $PROJECT_ROOT/configs/env_pypi.txt

srun python $PROJECT_ROOT/main.py fit \
    --config $PROJECT_ROOT/configs/base.yaml \
    --config $PROJECT_ROOT/configs/model/pmamba.yaml \
    --config $PROJECT_ROOT/configs/train/promptmamba/cmr25-cardiac.yaml 
