#!/bin/bash -l
#SBATCH -J VSSD-Recon_preproc
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                
#SBATCH --cpus-per-task=1                  
#SBATCH --mem=16GB                                                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

# Set PROJECT_ROOT and PYTHONPATH
export PROJECT_ROOT=/home/nicocarp/scratch/VSSD-Recon
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module purge

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

srun python prepare_h5_dataset_cmrxrecon.py \
    --input_matlab_folder $PROJECT_ROOT/raw_datasets/MICCAIChallenge2024/ChallengeData/MultiCoil \
    --output_h5_folder $PROJECT_ROOT/mridatasets/cmrxrecon2024/h5_dataset \
    --split_json $PROJECT_ROOT/configs/data_split/cmr24-cardiac.json \
    --year 2024
