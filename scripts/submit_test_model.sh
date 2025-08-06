#!/bin/bash -l
#SBATCH -J VSSD-Recon_test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4          
#SBATCH --mem=32GB                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

module --force purge

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Point to your project
export PROJECT_ROOT=/home/nicocarp/scratch/VSSD-Recon
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

python - <<'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# run your model file’s built-in test
srun python -u $PROJECT_ROOT/models/prompt_mamba.py
