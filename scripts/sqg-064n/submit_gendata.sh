#!/bin/bash

#SBATCH -J slurm_gendata
#SBATCH -o slurm_gendata.%j.out
#SBATCH -e slurm_gendata.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=spot
#SBATCH -t 120:00:00

python gendata.py
python concat.py
