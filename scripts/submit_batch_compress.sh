#!/bin/bash -l
#SBATCH -J cmrxrecon_compress
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

cd /home/nicocarp/scratch/VSSD-Recon/predict/cmr25-cardiac/test_VSSD-Recon_R1/

zip -r Submission.zip Submission