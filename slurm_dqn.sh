#!/bin/bash
$OUTPUTDIR=$1
# Need to add this to train_atari.py
# Run sbatch
for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test --mem=300G --partition=scu-gpu --gres=gpu:2 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash run_dqn.sh $RANDOMSEED $OUTPUTDIR"
done

sbatch --mem=5G --partition=scu-cpu --dependency=singleton --job-name=DQN_test --wrap="bash change_ownership.sh"

exit
