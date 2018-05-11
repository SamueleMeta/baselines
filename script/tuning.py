#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener

deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
deltas = map(str, deltas)
script = 'run_poispe_lqg'
path = 'tuning/adapoisnpe_linear'

commands = ['python3 sequential_experiment.py --script %s --path %s --delta %s' % (script,
                                                                                    path + '/delta_' + delta.replace('.',''),
                                                                                    delta)
                for delta in deltas]

Screener().run(commands, name='tuning')
for c in commands:
    print(c)
