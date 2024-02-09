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
import ipdb
import os

def average_trace(search_string):
    plt.figure()
    legend=[]
    agents=glob.glob(search_string)
    traces=[]
    min_data_len=100000
    for agent in agents:
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
        df=df.rolling(500).mean()
        if len(df)<min_data_len:
            min_data_len=len(df)  
        traces.append(df)
        
        #Get params
        _,_,_,_,seed,LR,BS=agent.split('_')
        BS,_=BS.split('.')
        #ipdb.set_trace()
        
        #plt.plot(df)
        
        #Subtract starting values
        baseline=df.mean(skipna=True)
        df=df-baseline
        
        if df.iloc[-1,0]>0.4:
            x=len(df)
            y=df.iloc[-1,0]
            string="LR: "+LR+",BS: "+BS
            plt.text(x,y,string,fontsize=10)    
        #ipdb.set_trace()
        if len(df)<600:
            plt.plot(df)
        plt.xlabel("Episodes")
        plt.ylabel("Episode Reward - Average Reward")
        
   
search_string="/athena/listonlab/scratch/anp4047/cohort1tuning/**/*.csv"
average_trace(search_string)
plt.savefig("/athena/listonlab/scratch/anp4047/cohort1tuning/ScaledReward.jpg")

