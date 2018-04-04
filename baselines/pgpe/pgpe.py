#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:13:18 2018

@author: matteo
"""

import numpy as np

def eval_trajectory(env, pol, gamma, task_horizon):
    ret = 0
    
    t = 0
    s = env.reset()
    done = False
    while not done and t<task_horizon:
        a = pol.act(s)
        s, r, done, _ = env.step(a)
        ret += gamma**t * r
        t+=1
        
    return ret
        

def learn(env, pol, gamma, step_size, batch_size, task_horizon, max_iterations, seed=None):
    if seed: env.seed(seed)
    
    #Learning iteration
    for it in range(max_iterations):
        rho = pol.eval_params() #Higher-order-policy parameters
        
        #Batch of episodes
        actor_params = []
        rets = []
        for ep in range(batch_size):
            #Initialize actor
            theta = pol.resample() #Sample actor parameters
            actor_params.append(theta)
            
            #Run episode
            ret = eval_trajectory(env, pol, gamma, task_horizon)
            rets.append(ret)
            
        print('%d: %f' % (it, np.mean(rets)))
        
        #Update higher-order policy
        grad = pol.eval_gradient(actor_params, rets)
        delta_rho = step_size * grad
        pol.set_params(rho + delta_rho)
            
    