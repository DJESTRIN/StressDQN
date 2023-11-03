#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:04:08 2023

@author: dje4001
"""
import gym
from gym.utils.play_custom import play
import os
directory_name='/athena/listonlab/scratch/anp4047/human_benchmark/'
subjectname=input('What is your first name?')
drop_folder=directory_name+subjectname+'/'
os.mkdir(drop_folder)
filename=drop_folder+'cumulative_rewards.csv'
env=gym.make('SpaceInvaders-v0')
env.env.game_difficulty=0
play(env,filename,zoom=4)




####
# add env.env.game_difficulty=1


