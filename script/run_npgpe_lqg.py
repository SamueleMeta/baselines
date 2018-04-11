#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""

import gym
import baselines.envs.continuous_cartpole
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.npgpe as npgpe
import baselines.envs.lqg1d

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

#Seeds: 0, 27, 62, 315, 640

def train(seed):
    DIR = '../results/npgpe/lqg/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = gym.make('LQG1D-v0')
    env.seed(seed)
    
    pol = PeMlpPolicy('pol',
                      env.observation_space,
                      env.action_space,
                      hid_size=64,
                      num_hid_layers=0,
                      use_bias=False,
                      standardize_input = True,
                      seed=seed)
    
    npgpe.learn(env,
              pol,
              gamma=0.99,
              step_size=1.,
              batch_size=100,
              task_horizon=500,
              max_iterations=500,
              use_baseline=True,
              step_size_strategy='norm',
              save_to=DIR,
              verbose=1)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.seed)