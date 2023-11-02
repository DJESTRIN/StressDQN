#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:04:08 2023

@author: dje4001
"""
import gym
from gym.utils.play2 import play,PlayPlot

if __name__=='__main__':
    env=gym.make('SpaceInvaders-v0')
    play(env,zoom=4)
    
    
    
    
    
    