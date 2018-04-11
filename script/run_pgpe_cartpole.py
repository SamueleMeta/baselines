#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""

import gym
import baselines.envs.continuous_cartpole
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.pgpe as pgpe

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

#Seeds: 0, 27, 62, 315, 640

def train(seed):
    DIR = '../results/pgpe/cartpole/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = gym.make('ContCartPole-v0')
    env.seed(seed)
    
    pol = PeMlpPolicy('pol',
                      env.observation_space,
                      env.action_space,
                      hid_size=64,
                      num_hid_layers=0,
                      use_bias=True,
                      standardize_input = True,
                      seed=seed)
    
    pgpe.learn(env,
              pol,
              gamma=0.99,
              step_size=0.1,
              batch_size=100,
              task_horizon=200,
              max_iterations=100,
              use_baseline=True,
              step_size_strategy='vanish',
              save_to=DIR,
              verbose=2)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.seed)