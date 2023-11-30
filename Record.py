#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import ipdb
import numpy as np
import os,glob

class Record():
    def __init__(self,seed,output_dir):
        self.seed=seed
        
        output_dir = output_dir+'/model_data/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # Delete previous data in output_dir
        previous_files = glob.glob(output_dir+'**/*.npy')
        for file in previous_files:
            os.remove(file)
            
        self.output_dir=output_dir
        
    def grab_w_n_b(self,agent,episode):
        """ Saves weight and bias information for each agent as a npy file """
        for name, param in agent.policy_network.named_parameters():
            #ipdb.set_trace()
            if 'fc' in name:
                output_name=self.output_dir+'/'+name.replace('.','_')+'/'
                if not os.path.exists(output_name):
                    os.mkdir(output_name)
                output_name=self.output_dir+'/'+name.replace('.','_')+'/'+str(self.seed)+'_'+str(episode)+"_"+name.replace('.','_')+'.npy'
                np.save(output_name,param.cpu().detach().numpy())
    
    def custom_sort(self,filename):
        try:
            _,_,_,_,number,_,_,_=filename.split('_')
        except:
            ipdb.set_trace()
        return int(number)
    
    def concat_w_n_b(self):
        """ Concatenates each episode's weights and biases for current agent """
        subfolders=glob.glob(self.output_dir+'/*')
        for folder in subfolders:
            for i,file in enumerate(sorted(glob.glob(folder+'/*.npy'),key=self.custom_sort)):
                if i==0:
                    fmat=np.load(file)
                    fmat=fmat[...,np.newaxis]
                    os.remove(file)
                else:
                    mat_oh=np.load(file)
                    mat_oh=mat_oh[...,np.newaxis]
                    os.remove(file)
                    fmat=np.concatenate((fmat, mat_oh),axis=-1) 
            output_name=folder+'/concat.npy'
            np.save(output_name,fmat)

    def unroll_episode(self, agent, state):
        """ Save neural activity and  """
        a=1
        
    
    
    
    
# To do --> 
    # Need weights at end of every episode for every neuron along with biases
    # Need recording of activity for each neuron to determien whether neuron is activated by threat
    #   Need Image at same time to determine when "threat" is present.