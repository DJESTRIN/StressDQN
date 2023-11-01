#!/bin/bash
#hylp1=$1
#hyp2=$2
#

#Activate correct conda environment
#source ~/.bashrc
conda activate StressDQN
module load cuda
nvidia-smi
python train_atari.py --random-seed 53

#python train_atari.py $hyperparameter1 $hyp2 $hyp3
