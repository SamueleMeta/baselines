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
WEIGHTS = list(map(lambda x: x.rstrip(' \r\n') if len(x.rstrip(' \r\n')) > 0 else None, """-0.03454712  0.27746254 -0.19657766 -0.09409703  0.06599652  0.04835585
 -0.24789995  0.29401935 -0.02833962 -0.00257341 -0.27647606 -0.15666797
 -0.33840969  0.47553276 -0.02822978  0.22682987 -0.15291106  0.38698434
  0.3819641   0.30835496  0.22088287 -0.48716542 -0.17617366  0.35655707
  0.13776421 -0.26616327  0.38614014 -0.48444382 -0.3289155  -0.06440699
  0.52631415 -0.12053841 -0.40676279 -0.26511784 -1.01352751 -0.70356369
  0.23377793  0.32676775  0.09885566 -0.20686346  0.16348059  0.24219478
  0.22427881  0.03345821  0.2986025  -0.23274482  0.05918973 -0.60387978
  0.39776098  0.43280944 -0.2237017   0.32023333  0.02867859  0.23139623
  0.10455188  0.17728067 -0.36209448 -0.67623584 -0.48278229 -0.70191282
  0.48574561 -0.55391487 -0.48961524  0.25405712 -0.23810904 -0.04587065
  0.18357824  0.1241943  -0.21981972 -0.24959933  0.18631034 -0.41167415
  0.0332289  -0.12784094 -0.02560754 -0.01399215 -0.00193327  0.03441378
  0.07617542 -0.05788043 -0.15373831 -0.24124652 -0.75378169  0.36635403
 -1.22758908 -0.50524694  0.04675505 -0.65675417  0.08621755 -0.10061
  0.10145083  0.05556994 -0.0321545  -0.12677491 -0.11808272  0.22338652
 -0.09570666""".replace('\n',' ').rstrip(' \r\n').split(' ')))
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
