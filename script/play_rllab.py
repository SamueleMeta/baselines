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
WEIGHTS = list(map(lambda x: x.rstrip(' \r\n') if len(x.rstrip(' \r\n')) > 0 else None, """-6.45358926e-02  2.72145804e-01 -4.85318013e-01 -8.06376243e-02
  1.74989187e-01 -2.23227808e-01 -2.98267727e-01  2.83639196e-01
  5.25926530e-02  6.25856643e-02 -2.71894920e-01 -2.92438154e-01
 -4.42991213e-01  4.09349364e-01  8.97584305e-02  3.29579285e-01
 -2.84920340e-01  3.86398815e-01  8.67764980e-02  3.15636771e-01
  3.31971196e-01 -4.66233859e-01 -4.93887694e-01  5.26173437e-01
  1.80919121e-01 -3.68760079e-01  4.54855203e-01 -4.42868762e-01
 -2.87294416e-01 -1.89357711e-01  5.06311976e-01 -8.51188660e-02
 -3.22454325e-01 -3.62088091e-01 -4.54256254e-01 -4.06686821e-01
  2.77973762e-01  4.01262319e-01  2.62879408e-01 -3.72356001e-01
  1.62679151e-01  2.23927198e-01  1.03566114e-01 -9.37338413e-02
  4.14875283e-01 -1.87690284e-01  5.92744153e-02 -4.41903256e-01
  4.06932484e-01  4.11498410e-01 -2.58009321e-01  2.61467877e-01
  3.28729810e-01  3.46728658e-01  1.00712266e-01  4.44401118e-01
 -3.88965603e-01 -4.84336563e-01 -4.62415343e-01 -5.47995497e-01
  3.65859107e-01 -4.84222162e-01 -5.42277903e-01  3.49075153e-01
  3.90402083e-04  1.84575928e-03  4.05871079e-03 -1.89278209e-03
  1.47705805e-03 -1.02044596e-04 -3.72434192e-02  2.04374115e-03
 -1.43568812e-04 -2.71378768e-03 -1.43661526e-03 -1.38386216e-04
  2.85787530e-03  1.10222967e-03 -3.27570099e-03 -1.41218913e-02
 -5.91672851e-02 -2.50208862e-02 -5.63610340e-02  5.68972645e-02
 -2.59013248e-01  1.04418206e-02  4.45938653e-02 -1.65858797e-01
  1.25868180e-03 -9.90512212e-02 -7.85307256e-02 -5.82946550e-03
 -8.97576227e-02 -9.88594439e-02  1.09536735e-01 -6.64527083e-02
  2.01050921e-03""".replace('\n',' ').rstrip(' \r\n').split(' ')))
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
