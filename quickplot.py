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

def average_trace(parent_dir):
    plt.figure()
    legend=[]
    for subdir in os.listdir(parent_dir):
        legend.append(subdir)
        search_string=os.path.join(parent_dir, subdir)+"/**/*.csv"
        agents = glob.glob(search_string)
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
        
        counter=0
        for trace in traces:
            trace_oh=trace[:min_data_len]
            if counter==0:
                trace_sum=trace_oh
            else:
                trace_sum+=trace_oh
            counter+=1

        trace_sum=trace_sum/counter
        plt.plot(trace_sum)
    plt.legend(legend,loc=(1.04,0))
    plt.title("Agent performance across difficulty")
   
search_string="/athena/listonlab/scratch/anp4047/cohort1/"
average_trace(search_string)

