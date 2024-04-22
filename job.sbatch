#!/bin/bash

#SBATCH -J zyg_Dst
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda env list
source /data2/ztwang/project/zyg/anaconda/install/bin/activate Swarm_Dst
conda env list

python test.py


#python CMB_test_xueyuan.py
