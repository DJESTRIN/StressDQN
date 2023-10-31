#!/bin/bash
#Grab hyperparameters
learning_rate=$1
batch_size=$2
output_dir=/athena/listonlab/scratch/dje4001/DQN/stress_dqn_tuning_results/
mkdir -p $output_dir

#Activate correct conda environment
source ~/.bashrc
conda activate StressDQN
python train_atari_tuning.py --learning_rate $learning_rate --batch_size $batch_size --output_dir $output_dir
