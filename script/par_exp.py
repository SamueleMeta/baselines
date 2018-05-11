#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener

delta = '0.9'
seeds = [662, 963, 100, 746, 236, 247, 689, 153, 947, 307, 42 , 950, 315, 545, 178]
path = 'may11/poisnpe'

commands = ['python3 run_poispe_lqg.py --seed %d --path %s --delta %s' % (seed,
                                                                                                   path,
                                                                                                   delta)
                for seed in seeds]

Screener().run(commands, name='lqg')
for c in commands:
    print(c)
