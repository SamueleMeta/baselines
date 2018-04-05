#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""

import gym
import baselines.envs.lqg1d
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.pgpe as pgpe

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

SEED = 0

env = gym.make('LQG1D-v0')

pol = PeMlpPolicy('pol',
                  env.observation_space,
                  env.action_space,
                  hid_size=2,
                  num_hid_layers=0,
                  use_bias=False,
                  standardize_input = True,
                  seed=SEED)

pgpe.learn(env,
          pol,
          gamma=1.,
          step_size=1e-3,
          batch_size=100,
          task_horizon=10,
          max_iterations=500,
          seed=SEED,
          save_to='./temp')