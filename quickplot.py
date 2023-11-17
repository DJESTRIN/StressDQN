#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:42:24 2023

@author: dje4001
"""

import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

search_string="/athena/listonlab/scratch/anp4047/cohort2_stresstest/**/**/*.csv"
agents = glob.glob(search_string)

counter=1
for agent in agents:
    if counter>10:
        counter=1
        
    if counter==1:
        plt.figure()
        
    array=np.loadtxt(agent,delimiter=",",dtype=str)
    
    #Build dataset
    data=array[:,0]
    data_converted=[]
    for value in data:
        new_value=str(value).replace('[','')
        data_converted.append(new_value)
    data_converted=np.asarray(data_converted)
    data=data_converted.astype(float)
    df = pd.DataFrame(data)
    df=df.rolling(100).mean()
    plt.plot(df)
        
    counter+=1