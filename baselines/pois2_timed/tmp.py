#!/usr/bin/env python3
import argparse
from baselines import bench, logger
import time, os, gym, logging
import tensorflow as tf
import numpy as np

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.pois2.cnn_policy import CnnPolicy
from baselines.pois2_timed import pois2
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

def train(env, max_iters, num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, gamma, seed, policy, max_offline_iters, njobs=1):

    # Declare env and created the vectorized env
    def make_env(seed=0):
        def _thunk():
            _env = make_atari(env)
            _env.seed(seed)
            return wrap_deepmind(_env)
        return _thunk
    parallel_env = VecFrameStack(SubprocVecEnv([make_env(i + seed) for i in range(njobs)], terminating=True), 4)

    obs = parallel_env.reset()
    print(obs.shape)
    print(obs[0, :2, :2, :])

    dummy_env = make_env(0)()
    for i in range(10):
        parallel_env.step_async([dummy_env.action_space.sample()] * njobs)
        obs, rew, done, _ = parallel_env.step_wait()
        print(obs[0, 74:84, :, :])

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    # Configure logging
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='logs', format_strs=['stdout', 'csv'], file_name=file_name)
    # Start training
    train(env=args.env,
          max_iters=args.max_iters,
          num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          delta=args.delta,
          gamma=args.gamma,
          seed=args.seed,
          policy=args.policy,
          max_offline_iters=args.max_offline_iters,
          njobs=args.njobs)


if __name__ == '__main__':
    main()
