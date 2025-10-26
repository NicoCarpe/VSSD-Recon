#!/bin/bash -l
#SBATCH -J VSSD-Recon_predict
#SBATCH --time=00-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

module --force purge

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Point to project
export PROJECT_ROOT=/home/nicocarp/scratch/VSSD-Recon
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module load StdEnv/2023
module load gcc/12.3
module load hdf5/1.14.2
module load cuda/12.2
module load cudnn/8.9.5.29
module load nccl/2.18.3
module load python/3.10

# enable H100 tensor-core matmuls (inside Python)
export PYTHONSTARTUP=$PROJECT_ROOT/.pystartup_matmul

# create a clean venv
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

python -m pip install --upgrade pip
python -m pip install --no-index -r $PROJECT_ROOT/configs/env_local.txt
python -m pip install -r $PROJECT_ROOT/configs/env_pypi.txt

# Run evaluation
python $PROJECT_ROOT/test/inference.py --input /input --output /output