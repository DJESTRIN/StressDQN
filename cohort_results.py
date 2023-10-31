#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:41:31 2023

@author: dje4001
"""
import pandas as pd
import os, glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import ipdb
from scipy import stats
import subprocess


class agent():
    def __init__(self,file_directory,group):
        self.array=np.loadtxt(file_directory,delimiter=",",dtype=str)
        self.skip=False
        self.group=group
        self.period1="First 200"
        self.period2="Last 200"
        
        if len(self.array)<1000:
            self.skip=True
            return 
        
        #Build dataset
        data=self.array[:,0]
        data_converted=[]
        for value in data:
            new_value=str(value).replace('[','')
            data_converted.append(new_value)
        data_converted=np.asarray(data_converted)
        self.data=data_converted.astype(float)
        
        data=self.array[:,1]
        data_converted=[]
        for value in data:
            new_value=str(value).replace(']','')
            data_converted.append(new_value)
        data_converted=np.asarray(data_converted)
        self.period=data_converted.astype(str)
        
        self.moving_average()

    def moving_average(self):
        #Reward values on trainable trials
        df=pd.DataFrame({'reward':self.data,'period':self.period})
        df=df['reward']
        self.exp_bl=df[0:200].mean()
        self.exp_fin=df[-200:].mean()
        self.exp_trace=df.rolling(50).mean()
        return

class Cohort():
    def __init__(self):
        self.exploitation_baseline=[]
        self.exploitation_final=[]
        self.exploitation_trace=[]
        self.groups=[]
        self.periods1=[]
        self.periods2=[]
        
    def grab_agent_data(self,agent):
        self.exploitation_trace.append(agent.exp_trace)
        self.exploitation_baseline.append(agent.exp_bl)
        self.exploitation_final.append(agent.exp_fin)
        self.groups.append(agent.group)
        self.periods1.append(agent.period1)
        self.periods2.append(agent.period2)
    
    def compare_exploitation(self):
        self.exploitation_baseline=np.asarray(self.exploitation_baseline)
        self.exploitation_final=np.asarray(self.exploitation_final)
        ipdb.set_trace()
        #Plot averages
        fig, ax=plt.subplots()
        for b,f in zip(self.exploitation_baseline,self.exploitation_final):
            plt.plot([0,1],[b,f],color='black',alpha=0.5)
        bl_mean,bl_error=np.mean(self.exploitation_baseline),np.std(self.exploitation_baseline)/np.sqrt(len(self.exploitation_baseline))
        f_mean,f_error=np.mean(self.exploitation_final),np.std(self.exploitation_final)/np.sqrt(len(self.exploitation_final))
        plt.errorbar(x=0,y=bl_mean,yerr=bl_error)
        plt.errorbar(x=1,y=f_mean,yerr=f_error)
        plt.scatter([0,1],[bl_mean,f_mean],s=50,color=['blue','orange'])
        plt.title('Average Reward for First 100 vs last 100 Exploitation Episodes')
        plt.ylabel('Rolling Average Reward')
        
        #stats
        stats.ttest_rel(self.exploitation_baseline,self.exploitation_final)
      
        #Plot average trace
        traces = np.zeros([len(self.exploitation_trace),len(max(self.exploitation_trace,key = lambda x: len(x)))])
        minimum_episode=10**5
        for i,j in enumerate(self.exploitation_trace):
            if len(j)<minimum_episode:
                minimum_episode=len(j)
            traces[i][0:len(j)] = j
        traces=traces[:,0:minimum_episode]
        average_trace=np.mean(traces,axis=0)
        error_trace=np.std(traces,axis=0)/np.sqrt(traces.shape[0])
        
        plt.figure()
        plt.plot(average_trace,linewidth=0.5)
        plt.fill_between(range(len(average_trace)), average_trace+error_trace, average_trace-error_trace, facecolor='blue', alpha=0.3)
        plt.title('Average Reward for Exploitation Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Rolling Average Reward')
        string="N: {} agents".format(traces.shape[0])
        plt.text(2000,3.5,string)
        plt.tight_layout()
        return
        
    def build_tall(self,output_path):
        #Average Data Tall
        reward_val=self.exploitation_baseline+self.exploitation_final
        groups=self.groups+self.groups
        period=self.periods1+self.periods2
        agents=list(range(len(self.groups)))+list(range(len(self.groups)))
        df=pd.DataFrame({'agent':agents,'group':groups,'period':period,'reward':reward_val})
        output_string=output_path+'cohort1_averages.csv'
        df.to_csv(output_string,index=False)
        
        #Trace Data Tall
        traces = np.zeros([len(self.exploitation_trace),len(max(self.exploitation_trace,key = lambda x: len(x)))])
        minimum_episode=10**5
        for i,j in enumerate(self.exploitation_trace):
            if len(j)<minimum_episode:
                minimum_episode=len(j)
            traces[i][0:len(j)] = j
        traces=traces[:,0:minimum_episode]
        traces=np.asarray(traces)
        
        DF=pd.DataFrame()
        for i,trace in enumerate(traces):
            df_oh=pd.DataFrame({'group':np.repeat(self.groups[i],traces.shape[1]),'agent':np.repeat(i, traces.shape[1]),'trace':trace})
            DF = pd.concat([DF, df_oh], ignore_index=True)
        
        output_string=output_path+'cohort1_traces.csv'
        DF.to_csv(output_string,index=False)

if __name__=='__main__':
    #Find csv files
    search_string='/athena/listonlab/scratch/anp4047/cohort1/random_agent/**/*seed*.csv'
    results=glob.glob(search_string)
    cohort=Cohort()
    for file_string in results:
        agent_oh=agent(file_string,'random')
        
        if agent_oh.skip:
            print('skipped this agent')
        else:
            cohort.grab_agent_data(agent_oh)  
    
    search_string='/athena/listonlab/scratch/anp4047/cohort1/dqn/*seed*.csv'
    results=glob.glob(search_string)
    for file_string in results:
        agent_oh=agent(file_string,'dqn')
        
        if agent_oh.skip:
            print('skipped this agent')
        else:
            cohort.grab_agent_data(agent_oh)
    
    cohort.build_tall('/athena/listonlab/scratch/anp4047/cohort1/')
    #subprocess.run("R cohort1_behavioral_results.r")

