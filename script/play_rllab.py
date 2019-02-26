'''
    Play N episodes and get run stats or render one episode for visualization
    using RLLab standards.
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict
import pickle as pkl
import argparse

# Framework imports
import gym
import tensorflow as tf
from time import sleep
import numpy as np
from tqdm import trange

# Self imports: utils
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from rllab.envs.normalized_env import normalize
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import get_env_type
from rllab.core.network import MLP
# Lasagne
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import lasagne.nonlinearities as NL
import lasagne.init as LI
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy
from baselines.pois import pois
from baselines.common.parallel_sampler import ParallelSampler

# ============================
# HARD CODED WEIGHTS
# ============================
WEIGHTS = list(map(lambda x: x.rstrip(' \r\n') if len(x.rstrip(' \r\n')) > 0 else None, """2.68388745e-02  2.40896120e-01 -1.43632561e-01  8.57850444e-02
  7.18332802e-02  1.91287257e-01 -1.97985497e-01  3.14035414e-01
 -3.60288705e-02  9.21816682e-03 -2.61736501e-01 -8.04133009e-02
 -3.33931540e-01  4.55943794e-01 -3.76463307e-02  2.09655513e-01
 -2.06461028e-01  4.04089753e-01  5.46158140e-01  2.56306841e-01
  2.90671206e-01 -5.14858886e-01 -1.46327347e-01  1.75610640e-01
  1.84322605e-01 -2.36563653e-01  4.78097962e-01 -4.64239368e-01
 -3.08522493e-01  1.81948691e-02  4.93772632e-01 -8.64737470e-02
 -4.05865649e-01 -4.06331619e-01 -1.06454323e+00 -1.04992489e+00
  2.62826873e-01  3.20349869e-01  2.28084424e-02 -1.75630575e-01
  1.26688236e-01  2.99736412e-01  2.18653890e-01 -7.69303288e-02
  4.03177244e-01 -3.46280233e-01  3.46067657e-02 -7.01872400e-01
  3.99350323e-01  4.01729486e-01 -1.27951745e-01  2.27275019e-01
 -3.41913289e-02  1.76373658e-01  1.24803696e-02  1.49259760e-01
 -3.46405490e-01 -6.34911105e-01 -3.48403840e-01 -7.64044707e-01
  3.63862821e-01 -5.98765918e-01 -5.19224583e-01  2.10470153e-01
 -2.28213432e-01 -4.55597882e-02  4.02373628e-01  1.36236381e-01
 -6.18128023e-01 -5.04876459e-01  2.07772620e-01 -5.54607153e-01
  2.64792583e-02 -8.37113649e-02 -3.46774855e-04 -4.00530471e-02
  3.96344997e-02  7.88681532e-02  8.46878847e-02 -1.95119592e-02
 -1.21212375e-01 -2.19521456e-01 -9.94053833e-01  6.48150713e-01
 -1.43036255e+00 -7.22559332e-01  1.04701469e-01 -6.87409829e-01
  3.98177641e-02 -9.96163293e-02  1.28133671e-01  1.92583406e-01
 -4.89822724e-02 -1.08208541e-01  4.82808574e-02  2.81504089e-01
 -1.84586284e-01""".replace('\n',' ').rstrip(' \r\n').split(' ')))
WEIGHTS = [w for w in WEIGHTS if w is not None]
WEIGHTS = list(map(float, WEIGHTS))
# ============================


def play_episode(env, pi, gamma, horizon=500):
    ob = env.reset()
    done = False
    reward = disc_reward = 0.0
    disc = 1.0
    timesteps = 0
    frames = []
    while not done and timesteps < horizon:
        a, _ = pi.get_action(ob)
        ob, r, done, _ = env.step(0)
        reward += r
        disc_reward += r * disc
        disc *= gamma
        timesteps += 1
        # TODO: add renders
    return {
        'reward': reward,
        'frames': frames
    }

def create_policy_and_env(env, seed, policy, weights):
    # Session
    sess = U.single_threaded_session()
    sess.__enter__()

    # Create env
    env_class = rllab_env_from_name(env)
    env = normalize(env_class())

    # Create policy
    obs_dim = env.observation_space.flat_dim
    action_dim = env.action_space.flat_dim
    if policy == 'linear':
        hidden_sizes = tuple()
    elif policy == 'simple-nn':
        hidden_sizes = [16]
    else:
        raise Exception('NOT IMPLEMENTED.')
    # Creating the policy
    mean_network = MLP(
                input_shape=(obs_dim,),
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=None,
                output_b_init=None,
                output_W_init=LI.Normal(),
            )
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=hidden_sizes,
        mean_network=mean_network
    )
    # Set the weights
    if weights is not None:
        raise Exception('TODO load pickle file.')
    else:
        weights = WEIGHTS
    policy.set_param_values(weights)

    print(policy.get_param_values())

    return env, policy

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--weights', type=str, default=None, help='Pickle weights file. If None, use hardcoded weights.')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--command', type=str, choices=['stats', 'render'])
    args = parser.parse_args()

    env, pi = create_policy_and_env(args.env, args.seed, args.policy, args.weights)

    N = 1 if args.command == 'render' else args.episodes
    rewards = []
    for i in trange(N):
        stats = play_episode(env, pi, args.gamma)
        rewards.append(stats['reward'])

    if args.command == 'render':
        pass # TODO save frames
    elif args.command == 'stats':
        print("Mean reward:", np.mean(rewards))
    else:
        raise Exception('Unrecognized command.')

if __name__ == '__main__':
    main()
