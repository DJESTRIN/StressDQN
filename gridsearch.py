# -*- coding: utf-8 -*-
"""
Grid search algorithm for DQN
"""
import numpy as np
import subprocess

#Learning Rate
decay = lambda x: 10**(-1*(x+1))
expon = lambda x: 10**(x+1)

LR = [decay(x) for x in range(2,8)] #1
batchsize=[8, 32, 64, 128] #1

for i in LR:
    for j in batchsize:
        command="sbatch --partition=scu-gpu,sackler-gpu --mem=10G --gres=gpu:1 --wrap='bash run_dqn_tuning.sh {} {}'".format(i,j)
        subprocess.run(command)
                
            





