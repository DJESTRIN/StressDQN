#!/bin/bash
#hylp1=$1
#hyp2=$2
#
JUNK=$1
RANDOMSEED=$2
OUTPUTDIR=$3
#Activate correct conda environment
#source ~/.bashrc

if [ "$JUNK" == "junk" ]; then
  JUNK="--junk"
else
  JUNK=""
fi

if [ -n "$OUTPUTDIR" ]; then
  OUTPUTDIR="--output-dir $OUTPUTDIR"
fi
source ~/.bashrc
conda activate StressDQN
module load cuda
nvidia-smi
python train_atari.py --random-seed $RANDOMSEED $OUTPUTDIR $JUNK

#python train_atari.py $hyperparameter1 $hyp2 $hyp3