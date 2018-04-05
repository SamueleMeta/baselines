#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""

import gym
import baselines.envs.mountain_car
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.pgpe as pgpe
import numpy as np
from baselines.interaction.features import RBF
from baselines.interaction.replay import replay
import itertools

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

SEED = 0
DIR = '../results/pgpe/mountain_car/05_04_' + str(SEED)
import os
if not os.path.exists(DIR):
    os.makedirs(DIR)

env = gym.make('CustomMountainCar-v0')
env.seed(SEED)

xs = np.linspace(env.min_position, env.max_position, 3)
ys = [-env.max_speed, -env.max_speed/3, env.max_speed/3, env.max_speed]
ks = [np.array([x,y]) for (x,y) in itertools.product(xs, ys)]
rbf = RBF(ks, 1.)

pol = PeMlpPolicy('pol',
                  ob_space=rbf.state_space(),
                  ac_space=env.action_space,
                  hid_size=4,
                  num_hid_layers=0,
                  use_bias=False,
                  standardize_input = True,
                  seed=SEED)

pgpe.learn(env,
          pol,
          gamma=0.95,
          step_size=1.,
          batch_size=10,
          task_horizon=40,
          max_iterations=50,
          feature_fun=rbf.feature_fun,
          use_baseline=True,
          step_size_strategy='norm',
          save_to=DIR)

replay(env, 
       pol, 
       gamma=0.95, 
       task_horizon=40,
       n_episodes=1,
       feature_fun=rbf.feature_fun,
       tau=0.1)