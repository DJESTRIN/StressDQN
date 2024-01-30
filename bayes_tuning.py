#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Opt for Hyperparameters. 
Optimize model via serial calls to slurm
Based off of https://machinelearningmastery.com/what-is-bayesian-optimization/
"""
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import ipdb

ndim=7 

# objective function
def objective(x, noise=0.1):
 noise = normal(loc=0, scale=noise)
 x1,x2,x3,x4,x5,x6,x7=x[0],x[1],x[2],x[3],x[4],x[5],x[6]
 return (x1**2 + x2**3 * (sin(5 * pi * x1*x3)**6.0)-3*x5**x6+x7) + noise
 
def gt(x):
 # noise = normal(loc=0, scale=noise)
 x1,x2,x3,x4,x5,x6,x7=x[0],x[1],x[2],x[3],x[4],x[5],x[6]
 return (x1**2 + x2**3 * (sin(5 * pi * x1*x3)**6.0)-3*x5**x6+x7) 

# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
    return model.predict(X, return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
 # calculate the best surrogate score found so far
 yhat, _ = surrogate(model, X)
 best = max(yhat)
 # calculate mean and stdev via surrogate function
 mu, std = surrogate(model, Xsamples)
 #ipdb.set_trace()
 #mu = mu[:, 0]
 # calculate the probability of improvement
 probs = norm.cdf((mu - best) / (std+1E-9))
 return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model,ndim):
 # random search, generate random samples
 Xsamples = random((100,ndim))
 Xsamples = Xsamples.reshape(len(Xsamples), ndim)
 # calculate the acquisition function for each sample
 #ipdb.set_trace()
 scores = acquisition(X, Xsamples, model)
 # locate the index of the largest scores
 ix = argmax(scores)
 return Xsamples[ix, :]
 
def get_x_data(n):
    x_all=[]
    for i in range(n):
        x_all.append(arange(0,1,0.001))
    return asarray(x_all).T

# plot real observations vs surrogate function
def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X[:,0], y)
    # line plot of surrogate function across domain
    #ipdb.set_trace()
    Xsamples = get_x_data(ndim)
    Xsamples = Xsamples.reshape(len(Xsamples), ndim)
    ysamples, _ = surrogate(model, Xsamples)
    yreal = asarray([gt(x_oh) for x_oh in Xsamples])
    #ipdb.set_trace()
    Xplotting=Xsamples[:,0]
    #Xplotting=sorted(Xplotting)
    pyplot.plot(Xplotting, ysamples)
    pyplot.plot(Xplotting, yreal)
    # show the plot
    pyplot.show()
 
# sample the domain sparsely with noise
X = random((100,ndim))

y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), ndim)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)

# plot before hand
plot(X, y, model)
#ipdb.set_trace()
# perform the optimization process
for i in range(400):
 # select the next point to sample
 x = opt_acquisition(X, y, model,ndim)
 # sample the point
 actual = objective(x)
 # summarize the finding
 #ipdb.set_trace()
 x=x.reshape(1,ndim)
 est, _ = surrogate(model, x)
 #print('>x=%.3f, f()=%3f, actual=%.3f' % (x[0], est, [[actual]]))
 # add the data to the dataset
 X = vstack((X, x))
 y = vstack((y, [[actual]]))
 # update the model
 model.fit(X, y)
 
# plot all samples and the final surrogate function
pyplot.figure()
plot(X, y, model)

# best result
ix = argmax(y)
#print('Best Result: x1=%.3f, x2=%.3f, y=%.3f' % (X[ix,0], X[ix,1], y[ix]))

# For first run, pull hyper parameters randomly.. 30?
# Run DQN with 30 random sets
# Run dependency calling this script
# Save a plot of rolling average DQN results for all current dqns. 
# Save DQN median reward value and hyperparameters in csv in common directory
# If csv files exist, load in hyper parameters, generate 30 sets of new parameters to test
# Run DQN with 30 random sets
# Run dependency calling this script










