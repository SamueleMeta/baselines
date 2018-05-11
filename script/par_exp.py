#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener

delta = '0.4'
seeds = [662, 963, 100, 746, 236, 247, 689, 153, 947, 307, 42 , 950, 315, 545, 178]
script = 'run_poispe_rllab_basic'
path = 'may11/poisnpe'
env = 'cartpole'

commands = ['python3 run_poispe_rllab_basic.py --seed %d --path %s --env %s --delta %s' % (seed,
                                                                                                   path,
                                                                                                   env,
                                                                                                   delta)
                for seed in seeds]

Screener().run(commands, name='cartpole')
for c in commands:
    print(c)
