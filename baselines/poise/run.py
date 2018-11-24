#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This script runs rllab or gym environments. To run RLLAB, use the format
    rllab.<env_name> as env name, otherwise gym will be used.
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict
# Framework imports
import gym
import tensorflow as tf
# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.bounded_mlp_policy import MlpPolicyBounded
from baselines.policy.cnn_policy import CnnPolicy
from baselines.poise import poise
from baselines.pois.parallel_sampler import ParallelSampler
# Import custom envs
import baselines.envs.lqg1d #registered at import as gym env

def get_env_type(env_id):
    #First load all envs
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)
    # Get env type
    env_type = None
    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    return env_type

def train(env, policy, horizon, seed, bounded_policy, njobs=1, **alg_args):

    #Prepare environment maker
    if env.startswith('rllab.'):
        # Get env name and class
        env_name = re.match('rllab.(\w+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)
        # Define env maker
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
            # Atari, custom env creation
            def make_env():
                _env = make_atari(env)
                return wrap_deepmind(_env)
        else:
            # Not atari, standard env creation
            def make_env():
                env_rllab = gym.make(env)
                return env_rllab

    # Prepare policy maker
    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    if policy == 'linear' or policy == 'nn':
        if bounded_policy == 'True':
            def make_policy(name, ob_space, ac_space):
                return MlpPolicyBounded(
                    name=name, ob_space=ob_space, ac_space=ac_space,
                    hid_size=hid_size, num_hid_layers=num_hid_layers,
                    gaussian_fixed_var=True, use_bias=False, use_critic=False,
                    hidden_W_init=tf.constant_initializer(-0.1),
                    max_mean=None,
                    min_mean=None,
                    max_std=None,
                    min_std=None,
                    std_init=1)
        else:
            def make_policy(name, ob_space, ac_space):
                return MlpPolicy(
                    name=name, ob_space=ob_space, ac_space=ac_space,
                    hid_size=hid_size, num_hid_layers=num_hid_layers,
                    gaussian_fixed_var=True, use_bias=False, use_critic=False,
                    hidden_W_init=tf.constant_initializer(-0.1))

    elif policy == 'cnn':
        def make_policy(name, ob_space, ac_space):
            return CnnPolicy(
                name=name, ob_space=ob_space, ac_space=ac_space,
                gaussian_fixed_var=True, use_bias=False, use_critic=False,
                hidden_W_init=tf.contrib.layers.xavier_initializer(),
                output_W_init=tf.contrib.layers.xavier_initializer())
    else:
        raise Exception('Unrecognized policy type.')

    #Prepare (sequential) sampler to generate ONE trajectory at a time
    sampler = None

    #Initialize
    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()
    set_global_seeds(seed)
    gym.logger.setLevel(logging.WARN)

    #Learn
    poise.learn(make_env, make_policy, horizon=horizon,
                sampler=sampler, **alg_args)

    #Close sampler in the end
    # sampler.close()

def main(args):
    #Command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='LQG1D-v0')
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--delta', type=float, default=0.3)#delta piccolo -> grande bonus
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--render_after', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--bounded_policy', type=bool, default='True')
    args = parser.parse_args(args)

    #Log file name
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name + '_%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())

    #Configure logger
    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)

    #Learn
    train(env=args.env,
          policy=args.policy,
          horizon=args.horizon,
          seed=args.seed,
          bounded_policy=args.bounded_policy,
          njobs=args.njobs,
          iw_norm=args.iw_norm,
          delta=args.delta,
          gamma=args.gamma,
          max_offline_iters=args.max_offline_iters,
          max_iters=args.max_iters,
          render_after=args.render_after)


if __name__ == '__main__':
    main(sys.argv[1:])
