#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener

seeds = "\'10 109 904 160 570\'"
#deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
deltas = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
deltas = map(str, deltas)
script = 'run_poispe_rllab_basic'
path = 'tuning/adapoisnpe_linear'
env = 'cartpole'

commands = ['python3 sequential_experiment.py --seeds %s --script %s --path %s --env %s --delta %s' % (seeds,
                                                                                                   script,
                                                                                                   path + '/delta_' + delta.replace('.',''),
                                                                                                   env,
                                                                                                   delta)
                for delta in deltas]

Screener().run(commands, name='tuning_half')
for c in commands:
    print(c)
