#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois_parallel
from baselines.pois.parallel_sampler import traj_segment_generator
import baselines.common.tf_util as U
import ast
import time
import tensorflow as tf

import sys
sys.path.append('/home/alberto/rllab')

from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.box2d.cartpole_env import CartpoleEnv

def train(num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, njobs):

    def make_env():
        env_rllab = CartpoleEnv()
        env = Rllab2GymWrapper(env_rllab)
        return env

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=0, num_hid_layers=0, gaussian_fixed_var=True, use_bias=True, use_critic=False)

    #Do this before instantiating the tf session!
    sampler = traj_segment_generator(policy_fn, make_env, num_episodes, horizon, True, n_workers=njobs, seed=seed)

    sess = U.make_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = make_env()

    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    pois_parallel.learn(env, policy_fn, num_episodes=num_episodes, iters=100,
               horizon=horizon, gamma=1., delta=delta, use_natural_gradient=natural,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, sampler=sampler)

    sampler.close()
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='sn')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='J')
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=4)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = 'CARTPOLE_iw_norm=%s_delta=%s_seed=%s_%s' % (args.iw_norm, args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='.', format_strs=['stdout', 'csv'], file_name=file_name)
    train(num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          delta=args.delta,
          seed=args.seed,
          njobs=args.njobs)

if __name__ == '__main__':
    main()
