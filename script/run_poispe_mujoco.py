#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys
sys.path.remove('/home/alberto/baselines')
sys.path.append('/home/alberto/baselines_ours')

from mpi4py import MPI
from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois, pois_clean
import baselines.common.tf_util as U
import ast
import time
import numpy as np

sys.path.append('/home/alberto/rllab')

from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.mujoco.swimmer_env import SwimmerEnv

def train(num_episodes, horizon, iw_method, iw_norm, natural, bound, ess, constraint, seed):

    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = SwimmerEnv()
    env = Rllab2GymWrapper(env)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=[100, 50, 25], num_hid_layers=3, gaussian_fixed_var=True, use_bias=True, use_critic=False)
    #env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    pois_clean.learn(env, policy_fn, num_episodes=num_episodes, iters=500,
               horizon=horizon, gamma=1., delta=0.49, use_natural_gradient=natural,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, ess_correction=ess)

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='sn')
    parser.add_argument('--natural', type=bool, default=True)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='d2')
    parser.add_argument('--ess', type=bool, default=False)
    parser.add_argument('--constraint', type=float, default=np.inf)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = 'pois_swimmer_rllab_%s_%s_nocorrection' % (args.iw_norm, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='.', format_strs=['stdout', 'csv'], file_name=file_name)
    train(num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          ess=args.ess,
          constraint=args.constraint,
          seed=args.seed)

if __name__ == '__main__':
    main()
