#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
#import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
from baselines.envs.lqg1d import LQG1D
from baselines.envs.continuous_cartpole import CartPoleEnv
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import sys

BATCH_SIZE = 20 # MINIMUM batch size (actual batch size in case of fixed horizon)
HORIZON = 100 # MAXIMUM horizon
ITERATIONS = 100

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=1, num_hid_layers=0,gaussian_fixed_var=True,use_bias=False)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, batch_size = BATCH_SIZE, 
                   task_horizon = HORIZON, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    import argparse
    task_id = 'ContCartPole-v0'
    #task_id = 'LQG1D-v0'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=task_id)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(ITERATIONS*BATCH_SIZE*HORIZON))
    args = parser.parse_args()
    logger.configure(dir='.',format_strs=['stdout','csv'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
