#!/bin/bash
OUTPUTDIR=$1 #real
# Need to add this to train_atari.py
# Run sbatch
CURRENTEPOCTIME=`date +%s`
RANDOMSEED=$(($CURRENTEPOCTIME + $i))
sbatch --job-name=DQN_test --mem=100G --partition=scu-gpu --gres=gpu:1 --reservation=listonlab --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash run_dqn_Anthony.sh NoJunk not_random_agent dif_test_off $RANDOMSEED $OUTPUTDIR"

sbatch --mem=5G --partition=scu-cpu --dependency=singleton --job-name=DQN_test --wrap="bash change_ownership.sh"

exit




# Settings for run_dqn.sh script
# junk    -- will create junk agents
# random_agent -- will create random agents
# dif_test_on -- will create a difficulty test simulating stress
