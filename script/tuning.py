#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener

deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
deltas = map(str, deltas)
path = 'tuning_network'
seed = 109

commands = ['python3 run_multi_poispe_rllab_cartpole.py --path %s --delta %s --seed %d' % (path + '/delta_' + delta.replace('.',''),
                                                                                    delta, seed)
                for delta in deltas]

Screener().run(commands, name='tuning_network')
for c in commands:
    print(c)
