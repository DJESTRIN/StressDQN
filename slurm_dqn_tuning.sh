#!/bin/bash
OUTPUTDIR=$1 #real

for i in {1..10};
do
	for k in {1..10};
	do
		BS=$(($k*4))
		echo Batchsize equals $BS
		LR=$((10**$i))
		LR=$(echo  "1 / $LR" | bc -l)
		echo Learning rate equals $LR
		CURRENTEPOCTIME=`date +%s`
   		RANDOMSEED=$(($CURRENTEPOCTIME + $i + $k))
		echo Random seed equals $RANDOMSEED
   		sbatch --job-name=DQN_test --mem=10G --partition=scu-gpu --gres=gpu:1 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash run_dqn.sh NoJunk not_random notdiftest $RANDOMSEED $OUTPUTDIR $LR $BS"
	done
done

sbatch --mem=5G --partition=scu-cpu --dependency=singleton --job-name=DQN_test --wrap="bash change_ownership.sh"

exit

