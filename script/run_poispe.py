#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""

import gym
import baselines.envs.continuous_cartpole
import baselines.envs.lqg1d
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.poisnpe as poisnpe
import baselines.pgpe.poispe as poispe
import numpy as np

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

envs = {'cartpole': 'ContCartPole-v0',
        'lqg': 'LQG1D-v0',
        }

algos = {'poisnpe': poisnpe,
         'poispe': poispe,
        }

horizons = {'cartpole': 200,
            'lqg': 500,
            }

rews = {'cartpole': 10,
      'lqg': 28.8,
      }

iters = {'cartpole': 100,
         'lqg': 100,
         }

#Seeds: 107, 583, 850, 730, 808

def train(seed, env_name, algo_name, stop_sigma, gamma):
    DIR = 'temp/'
    #DIR = '../results/' + algo_name + '/z/' + env_name + '/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = gym.make(envs[env_name])
    env.seed(seed)
    horizon = horizons[env_name]
    #rmax = sum([rews[env_name]*gamma**i for i in range(horizon)])
    rmax = None #Empirical
    
    pol_maker = lambda name: PeMlpPolicy(name,
                      env.observation_space,
                      env.action_space,
                      hid_layers=[],
                      diagonal=True,
                      use_bias=False,
                      standardize_input=True,
                      seed=seed)
    
    algos[algo_name].learn(env,
              pol_maker,
              gamma=gamma,
              batch_size=100,
              task_horizon=horizon,
              max_iterations=100,
              save_to=DIR,
              verbose=2,
              feature_fun=np.ravel,
              correct_ess=True,
              normalize=True,
              stop_sigma=stop_sigma,
              max_offline_ite=100,
              max_search_ite=30,
              bound_name='z',
              rmax=rmax)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--stop', help='Stop sigma?', type=int, default=0)
    parser.add_argument('--gamma', help='Stop sigma?', type=float, default=0.99)
    parser.add_argument('--algo', help='Algorithm', type=str, default='poisnpe')
    parser.add_argument('--env', help='Environment (RL task)', type=str, default='cartpole')
    args = parser.parse_args()
    train(args.seed, args.env, args.algo, args.stop, args.gamma)
