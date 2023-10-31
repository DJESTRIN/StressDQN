#!/bin/bash

# Need to add this to train_atari.py
#$input=$1
#$output=$2

# Run sbatch
srun --job-name=DQN_test_Space_Invaders --mem=300G --partition=scu-gpu --gres=gpu:2 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --pty bash run_dqn.sh
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
