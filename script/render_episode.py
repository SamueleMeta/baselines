#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This file is intended to be used to render an episode for a given policy
    and a given environment.

    TODO:
    - Understand rendering in RLLAB
    - Understand rendering in remote server

'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict
import pickle as pkl

# Framework imports
import gym
import tensorflow as tf
from time import sleep

# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import get_env_type
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy
from baselines.pois import pois
from baselines.common.parallel_sampler import ParallelSampler

def play_episode(env, pi, gamma, filename='render.pkl'):

    ob = env.reset()
    done = False
    reward = 0
    disc_reward = 0
    disc = 1.0
    timesteps = 0
    renders = []
    while not done:
        ac, vpred = pi.act(True, ob)
        print("ACTION:", ac)
        ob, r, done, _ = env.step(ac)
        reward += r
        disc_reward += r * disc
        disc *= gamma
        timesteps += 1
        rend = env.render(mode='rgb_array', close=False)
        renders.append(rend)
        sleep(0.05)
    print("Finished episode.")
    print("Total timesteps:", timesteps)
    print("Total reward:", reward)

    pkl.dump(renders, open(filename, 'wb'))

def create_policy_and_env(env, seed, policy, policy_file):
    # Session
    sess = U.single_threaded_session()
    sess.__enter__()

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
                             hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=False,
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

    #weights = pkl.load(open(policy_file, 'rb'))
    # TMP overriding weights
    #weights = [-0.19337249, -0.12103618, 0.00849289, -0.1105529, -3.6525128] # TRPO
    #weights = [-0.5894, -0.2585, -0.0137, -0.2464, -0.2788] # POIS
    #weights = list(map(float, ['-0.5807', '-0.3046', '-0.0127', '-0.3045', '-0.7427']))
    weights = list(map(float, ['-0.3637', '0.4266', '0.4686', '-0.1819', '-0.3876', '0.3364', '-0.4700', '0.4523', '0.5114', '-0.4117', '0.3854', '-0.2999', '-0.1184', '-0.2932', '-0.3350', '-0.5315', '-0.1599', '0.2814', '-0.4896', '-0.1338', '0.5249', '0.4585', '-0.2125', '0.1202', '0.4842', '0.4239', '0.5364', '-0.4462', '-0.3725', '0.1594', '0.2518', '0.3391', '0.0329', '0.2546', '-0.4698', '0.1880', '-0.3797', '0.3734', '0.2662', '-0.1021', '0.1367', '0.5337', '0.0304', '-0.1175', '0.0153', '-0.4145', '0.3839', '0.2762', '0.0015', '0.2586', '-0.2759', '-0.4853', '0.3772', '-0.4575', '0.4328', '0.1659', '-0.0043', '0.4972', '0.0466', '-0.0623', '-0.1515', '-0.5166', '0.2875', '0.5312', '0.5631', '0.0962', '-0.3479', '0.1646', '-0.2556', '-0.5216', '0.3815', '-0.4457', '0.2350', '0.0673', '0.0299', '-0.3655', '0.1201', '0.3468', '0.0051', '0.1402', '0.0405'] ))
    pi.set_param(weights)

    return env, pi

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
    play_episode(env, pi, args.gamma, args.output)

if __name__ == '__main__':
    main()
