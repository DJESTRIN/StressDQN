#!/bin/bash
JUNK=$1
RANDOM_AGENT=$2
DIF_TEST=$3
RANDOMSEED=$4
OUTPUTDIR=$5

#Determine type of run
if [ "$JUNK" == "junk" ]; then
  JUNK="--junk"
else
  JUNK=""
fi

if [ "$RANDOM_AGENT" == "random_agent" ]; then
  RANDOM_AGENT="--random-choice"
else
  RANDOM_AGENT=""
fi

if [ "$DIF_TEST" == "dif_test_on" ]; then
  DIF_TEST="--difficulty-test"
else
  DIF_TEST=""
fi

if [ -n "$OUTPUTDIR" ]; then
  OUTPUTDIR="--output-dir $OUTPUTDIR"
fi

#Execute python command
source ~/.bashrc
conda activate StressDQN
module load cuda
nvidia-smi
python train_atari.py --random-seed $RANDOMSEED $OUTPUTDIR $JUNK $RANDOM_AGENT $DIF_TEST
