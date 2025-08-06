#!/bin/bash -l
#SBATCH -J cmrxrecon_postproc
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

# load modules
module purge
module load StdEnv/2023
module load matlab/2024b.1

# set project root
export PROJECT_ROOT=/home/nicocarp/scratch/VSSD-Recon

matlab -singleCompThread -batch "\
  cd('$PROJECT_ROOT/support/CMRxRecon2025/CMRxReconDemo'); \
  run('mainRun4Ranking_2025.m'); \
"
