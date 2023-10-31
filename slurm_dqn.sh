#!/bin/bash

# Need to add this to train_atari.py
#$input=$1
#$output=$2

# Run sbatch
sbatch --job-name=DQN_test --mem=300G --partition=scu-gpu --gres=gpu:2 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="bash ~/StressDQN/run_dqn.sh"
exit

# 500 jobs total
# seed ===> this can be changed based on time
# LR ==> 
# BS ==>
# DF ==> 
# RBS ==>
# Steps ==>
# LearningStarts ==>
# EpislonGreedy==>
#
#
