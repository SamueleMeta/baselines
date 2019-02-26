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
WEIGHTS = list(map(lambda x: x.rstrip(' \r\n') if len(x.rstrip(' \r\n')) > 0 else None, """0.15953386  0.10822787  0.30392429 -0.19635692 -0.47422308  0.14848398
 -0.19973465 -0.36574065  0.40281759 -0.08021425 -0.01819268 -0.06295348
  0.22438566 -0.02605306  0.30656943  0.61153343  0.21480066 -0.51247014
 -0.23002151 -0.30995863  0.2287879   0.08066655 -0.32371414  0.56897408
 -0.18204293 -0.50848639  0.43094811  0.70952242  0.38075873  0.4604741
  1.02463092  0.17539713  0.81328516 -0.35254996  0.07754536  0.77557677
  0.34065107 -0.48293102  0.04958217 -0.14484813  0.45851386  0.1602454
  0.41681276  0.30192835  0.58199219  0.73821455 -0.44703031 -0.17168907
 -0.30993005 -0.36565573  0.7119807   0.1930647   0.37343778  0.83623938
 -0.61747588  0.52922941  0.16388354  0.03601739 -0.34726328  0.00309361
 -0.41042698 -0.31262208 -0.17347692 -0.2047279  -0.10537838 -0.08110084
 -0.03735622  0.03708094 -0.02780432 -0.06436927 -0.10838084  0.10670034
 -1.28798893  0.49476737  0.04617231 -0.41995246  0.09265398 -0.02882002
  0.50526057  0.30239238 -0.50592869 -0.42284231  0.36936024  0.57067307
  0.10089432  0.46859915  0.1611406  -0.62135622 -1.51016674  1.11468115
  0.09764898 -1.17192074  0.2054546  -0.23601387  0.69082661  0.20433299
 -0.32798772""".replace('\n',' ').rstrip(' \r\n').split(' ')))
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
