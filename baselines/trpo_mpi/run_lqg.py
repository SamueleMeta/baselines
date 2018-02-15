#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
#from baselines.trpo_mpi import trpo_mpi
from baselines.trpo_mpi import ours
from baselines.envs.lqg1d import LQG1D
import sys

def train(num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = LQG1D()
    env.horizon = 20
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=0, num_hid_layers=0, gaussian_fixed_var=True, use_bias=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    ours.learn(env, policy_fn, timesteps_per_batch=4000, iters=10,
        max_timesteps=num_timesteps, gamma=0.99, delta=0.2, N=200)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=100*20*200)
    args = parser.parse_args()
    logger.configure(dir='.', format_strs=['stdout', 'csv'])
    train(num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()