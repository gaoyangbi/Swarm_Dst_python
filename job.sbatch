#!/bin/bash

#SBATCH --partition=hpib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00


cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate landlab

python CMB_test.py
