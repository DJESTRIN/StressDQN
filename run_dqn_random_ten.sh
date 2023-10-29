# !/bin/bash

for i in {1..10}
do
    CURRENTEPOCTIME=`date +"%s"`
    mkdir seed${CURRENTEPOCTIME}
    bash run_dqn.sh --random-seed ${CURRENTEPOCTIME} > seed${CURRENTEPOCTIME}/output.txt
done
