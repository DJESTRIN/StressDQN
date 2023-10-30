#!/bin/bash
#hylp1=$1
#hyp2=$2
#
seedCounter=$1

#Activate correct conda environment
source ~/.bashrc
conda activate StressDQN
module load cuda
nvidia-smi
python train_atari_sbatch.py --random-seed $seedCounter

#python train_atari.py $hyperparameter1 $hyp2 $hyp3
