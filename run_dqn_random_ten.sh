# !/bin/bash

for i in {1..10}
do
    source ~/.bashrc
    conda activate StressDQN
    module load cuda
    nvidia-smi
    CURRENTEPOCTIME=`date +"%s"`
    mkdir seed${CURRENTEPOCTIME}
    chmod u+w seed${CURRENTEPOCTIME}
    python train_atari.py --random-seed ${CURRENTEPOCTIME} > seed${CURRENTEPOCTIME}/output.txt
done
