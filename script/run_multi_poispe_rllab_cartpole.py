#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""
import sys
sys.path.append('/home/alberto/rllab')
sys.path.append('/home/matteo/rllab')

from baselines.policy.neuronwise_pemlp_policy import MultiPeMlpPolicy
import baselines.pgpe.neuronwise_poisnpe_par as multipoisnpe
from baselines.pgpe.parallel_sampler import ParallelSampler
import numpy as np

from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.box2d.cartpole_env import CartpoleEnv

import baselines.common.tf_util as U
from baselines.common import set_global_seeds

def train(seed, shift, normalize, use_rmax, use_renyi, path):
    index = int(str(int(shift)) + str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
    DIR = path + '/multipoisnpe/bound_' + str(index) + '/cartpole_rllab/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    def env_maker():
        env = CartpoleEnv()
        env = Rllab2GymWrapper(env)
        return env
        
    pol_maker = lambda name, observation_space, action_space: MultiPeMlpPolicy(name,
                      observation_space,
                      action_space,
                      hid_layers=[10,5,2],
                      use_bias=True,
                      seed=seed)
    
    batch_size = 100
    gamma = 1.
    horizon = 500
    njobs = -1
    sampler = ParallelSampler(env_maker, pol_maker, gamma, horizon, np.ravel, batch_size, njobs, seed)
    
    sess = U.make_session()
    sess.__enter__()

    set_global_seeds(seed)
 
    
    multipoisnpe.learn(env_maker,
              pol_maker,
              sampler,
              gamma=gamma,
              initial_batch_size=batch_size,
              task_horizon=horizon,
              max_iterations=500,
              save_to=DIR,
              verbose=1,
              feature_fun=np.ravel,
              normalize=normalize,
              use_rmax=use_rmax,
              use_renyi=use_renyi,
              max_offline_ite=20,
              max_search_ite=1,
              delta=1.,
              shift=shift,
              use_parabola=False)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='save here', type=str, default='temp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--shift', help='Normalize return?', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize weights?', type=int, default=1)
    parser.add_argument('--use_rmax', help='Use Rmax in bound (or var)?', type=int, default=1)
    parser.add_argument('--use_renyi', help='Use Renyi in ESS (or weight norm)?', type=int, default=0)
    args = parser.parse_args()
    train(args.seed,
          args.shift,
          args.normalize, 
          args.use_rmax,
          args.use_renyi,
          args.path)
