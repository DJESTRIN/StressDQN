#!/bin/bash
#hyp1=$1
#hyp2=$2
#

#Activate correct conda environment
source ~/.bashrc
conda activate StressDQN
python train_atari.py 

#python train_atari.py $hyperparmeter1 $hyp2 $hyp3
