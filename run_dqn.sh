#!/bin/bash
#hylp1=$1
#hyp2=$2
#
RANDOMSEED=$1
OUTPUTDIR=$2
#Activate correct conda environment
#source ~/.bashrc

if [ -n "$OUTPUTDIR" ]; then
  OUTPUTDIR="--output-dir $OUTPUTDIR"
fi

conda activate StressDQN
module load cuda
nvidia-smi
python train_atari.py --random-seed $RANDOMSEED $OUTPUTDIR

#python train_atari.py $hyperparameter1 $hyp2 $hyp3