#!/bin/bash

# Need to add this to train_atari.py
#$input=$1
#$output=$2

# Run sbatch
for i in {1..10}
do
    sbatch --job-name=DQN_test --mem=50G --partition=scu-gpu,sackler-gpu --gres=gpu:1 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash ~/StressDQN/run_junk_dqn.sh $i"
done
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
