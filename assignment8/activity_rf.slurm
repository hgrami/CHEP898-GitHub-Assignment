#!/bin/bash
#SBATCH --job-name=activity_rf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=activity_rf_%j.out
#SBATCH --error=activity_rf_%j.err
#SBATCH --mail-user=lzn538@usask.ca
#SBATCH --mail-type=END,FAIL

cd /home/lzn538

module load StdEnv/2023 r/4.5.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun Rscript activity_rf.R
