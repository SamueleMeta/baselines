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

SEED = 1
DIR = '../results/pgpe/cartpole/05_04_' + str(SEED)
import os
if not os.path.exists(DIR):
    os.makedirs(DIR)

env = gym.make('ContCartPole-v0')
env.seed(SEED)

pol = PeMlpPolicy('pol',
                  env.observation_space,
                  env.action_space,
                  hid_size=8,
                  num_hid_layers=2,
                  use_bias=True,
                  standardize_input = True,
                  seed=SEED)

pgpe.learn(env,
          pol,
          gamma=0.99,
          step_size=0.01,
          batch_size=100,
          task_horizon=100,
          max_iterations=500,
          use_baseline=True,
          step_size_strategy=None,
          save_to=DIR,
          verbose=1)