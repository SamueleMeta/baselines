#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This file is intended to be used to render an episode for a given policy
    and a given environment.
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict
import pickle as pkl

# Framework imports
import gym
import tensorflow as tf
from time import sleep
import numpy as np

# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
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

def play_episode(env, pi, gamma, filename='render.pkl', horizon=500):

    ob = env.reset()
    done = False
    reward = 0
    disc_reward = 0
    disc = 1.0
    timesteps = 0
    renders = []
    while not done and timesteps < horizon:
        ac, vpred = pi.act(True, ob)
        #print("ACTION:", ac)
        ob, r, done, _ = env.step(ac)
        reward += r
        disc_reward += r * disc
        disc *= gamma
        timesteps += 1
        #rend = env.render(mode='rgb_array', close=False)
        #renders.append(rend)
        #sleep(0.01)
    print("Finished episode.")
    print("Total timesteps:", timesteps)
    print("Total reward:", reward)

    #pkl.dump(renders, open(filename, 'wb'))
    return reward

def create_policy_and_env(env, seed, policy, policy_file):
    # Session
    sess = U.single_threaded_session()
    sess.__enter__()
    '''
    # Create the environment
    if env.startswith('rllab.'):
        # Get env name and class
        env_name = re.match('rllab.(\S+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)
        # Define env maker
        def make_env():
            env_rllab = env_rllab_class()
            _env = Rllab2GymWrapper(env_rllab)
            return _env
        # Used later
        env_type = 'rllab'
    else:
        # Normal gym, get if Atari or not.
        env_type = get_env_type(env)
        assert env_type is not None, "Env not recognized."
        # Define the correct env maker
        if env_type == 'atari':
            # Atari, custom env creation
            def make_env():
                _env = make_atari(env)
                return wrap_deepmind(_env)
        else:
            # Not atari, standard env creation
            def make_env():
                env_rllab = gym.make(env)
                return env_rllab
    env = make_env()
    env.seed(seed)
    ob_space = env.observation_space
    ac_space = env.action_space
    '''
    env_class = rllab_env_from_name(env)
    env = normalize(env_class())



    '''
    # Make policy
    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'simple-nn':
        hid_size = [16]
        num_hid_layers = 1
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3
    # Temp initializer
    policy_initializer = U.normc_initializer(0.0)
    if policy == 'linear' or policy == 'nn' or policy == 'simple-nn':
        def make_policy(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=True, use_critic=False,
                             hidden_W_init=policy_initializer,
                             output_W_init=policy_initializer)
    elif policy == 'cnn':
        def make_policy(name, ob_space, ac_space):
            return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=policy_initializer,
                         output_W_init=policy_initializer)
    else:
        raise Exception('Unrecognized policy type.')
    pi = make_policy('pi', ob_space, ac_space)
    # Load policy weights from file
    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]
    set_parameter = U.SetFromFlat(var_list)
    '''
    obs_dim = env.observation_space.flat_dim
    action_dim = env.action_space.flat_dim
    policy_init = 'zeros'
    # Policy initialization
    if policy_init == 'zeros':
        initializer = LI.Constant(0)
    elif policy_init == 'normal':
        initializer = LI.Normal()
    else:
        raise Exception('Unrecognized policy initialization.')

    # Setting the policy type
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
                hidden_sizes=[16],
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=None,
                output_b_init=None,
                output_W_init=initializer,
            )
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=[16],
        mean_network=mean_network
    )

    #weights = pkl.load(open(policy_file, 'rb'))
    # TMP overriding weights
    #weights = [-0.19337249, -0.12103618, 0.00849289, -0.1105529, -3.6525128] # TRPO
    #weights = [-0.5894, -0.2585, -0.0137, -0.2464, -0.2788] # POIS
    #weights = list(map(float, ['-0.5807', '-0.3046', '-0.0127', '-0.3045', '-0.7427']))
    weights = list(map(lambda x: x.rstrip(' \r\n') if len(x.rstrip(' \r\n')) > 0 else None, """0.02483223 -0.17645608  0.77450023  0.54770311  0.33464952 -0.29827444
 -0.62524864  0.46413191 -0.31990006 -0.32972003  0.38753632 -0.15170416
 -0.43518174 -0.15718946  0.19542838 -0.02774486  0.13546377 -0.18621497
  0.18444675  0.774653    0.19710147 -0.20958339  0.15098953  0.42278248
 -0.53121678 -0.33369185 -0.04331141 -0.2140371   0.27077572  0.58111134
  0.34637848  0.56956591  0.45061681 -0.15826946 -1.06925573 -0.39311001
 -0.35695692  0.14414285 -1.25332428 -0.24016012  0.17774961  0.23973508
 -0.65415459  1.53059934 -0.71953132  1.79764386  0.18561774  1.4640445
 -0.1625999   0.0606595  -0.22058723 -0.34247517  0.46232139  0.07013392
 -0.32074007  0.14488911  0.1123158   0.28914362  0.6727726  -0.58491444
  0.35895434  1.32873906 -0.0708237  -0.05147256  0.01689644  0.38244615
  0.10005984  0.71253728 -0.18824528 -0.15552894 -0.05634595  0.3517145
  0.20900426 -0.19631462 -0.03828797  0.08125694 -0.22894259 -0.08030374
  0.59522035 -0.1752422  -0.40809067  1.62409963 -1.39307047  0.81438794
 -0.54068521  0.19321547 -1.65661292  0.3264788   0.46482921 -0.01649974
 -0.79186757 -1.3378886  -0.57094913 -1.57079733 -1.78056839  1.05324632
 -2.14386428""".rstrip(' \r\n').split(' ')))
    weights = [w for w in weights if w is not None]
    weights = list(map(float, weights))

    print(weights)
    #pi.set_param(weights)

    return env, policy

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--policy_file', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='render.pkl')
    args = parser.parse_args()

    env, pi = create_policy_and_env(args.env, args.seed, args.policy, args.policy_file)
    rs = []
    for i in range(100):
        r = play_episode(env, pi, args.gamma, args.output)
        rs.append(r)
    print(sum(rs) / len(rs))

if __name__ == '__main__':
    main()
