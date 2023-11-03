#!/bin/bash
#hylp1=$1
#hyp2=$2
#
$input=$1
#Activate correct conda environment
#source ~/.bashrc
conda activate StressDQN
module load cuda
nvidia-smi
python train_atari.py --random-seed $input

#python train_atari.py $hyperparameter1 $hyp2 $hyp3
