#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois
import baselines.common.tf_util as U
import time
import os
import tensorflow as tf
from baselines.pois.parallel_sampler import ParallelSampler

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

# Create experiment
ex = Experiment('POIS')
# Set a File Observer
ex.observers.append(FileStorageObserver.create('sacred_runs'))
ex.observers.append(SlackObserver.from_config('../../configs/slack.json'))

@ex.config
def custom_config():
    seed = 0
    env = 'cartpole'
    num_episodes = 100
    max_iters = 500
    horizon = 500
    iw_method = 'is'
    iw_norm = 'none'
    natural = False
    file_name = 'progress'
    logdir = '.'
    bound = 'max-d2'
    delta = 0.99
    njobs = -1
    policy = 'nn'
    max_offline_iters = 10
    gamma = 1.0
    center = True
    clipping = False
    entropy = 'none'
    positive_return = False
    # ENTROPY can be of 4 schemes:
    #    - 'none'
    #    - 'step:<height>:<duration>': step function which is <height> tall for <duration> iterations
    #    - 'lin:<max>:<min>': linearly decreasing function from <max> to <min> over all iterations, clipped to 0 for negatives
    #    - 'exp:<height>:<scale>': exponentially decreasing curve <height> tall, use <scale> to make it "spread" more
    # Create the filename
    if file_name == 'progress':
        file_name = '%s_iw=%s_bound=%s_delta=%s_gamma=%s_center=%s_entropy=%s_seed=%s_%s' % (env.upper(), iw_method, bound, delta, gamma, center, entropy, seed, time.time())
    else:
        file_name = file_name

def train(env, num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, policy, max_offline_iters, gamma, center_return, clipping=False, njobs=1, entropy='none', max_iters=500, positive_return=False):

    def make_env():
        _env = gym.make(env)
        return _env

    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    def make_policy(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=tf.contrib.layers.xavier_initializer(),
                         output_W_init=tf.contrib.layers.xavier_initializer())

    sampler = ParallelSampler(make_policy, make_env, num_episodes, horizon, True, n_workers=njobs, seed=seed)

    affinity = len(os.sched_getaffinity(0))
    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)

    pois.learn(make_env, make_policy, n_episodes=num_episodes, max_iters=max_iters,
               horizon=horizon, gamma=gamma, delta=delta, use_natural_gradient=natural,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, save_weights=True, sampler=sampler,
               center_return=center_return, render_after=None, max_offline_iters=max_offline_iters,
               clipping=clipping, entropy=entropy, positive_return=positive_return)

    sampler.close()

@ex.automain
def main(seed, env, num_episodes, horizon, iw_method, iw_norm, natural, file_name, logdir, bound, delta,
            njobs, policy, max_offline_iters, gamma, center, clipping, entropy, max_iters, positive_return, _run):

    logger.configure(dir=logdir, format_strs=['stdout', 'csv', 'tensorboard', 'sacred'], file_name=file_name, run=_run)
    train(env=env,
          num_episodes=num_episodes,
          horizon=horizon,
          iw_method=iw_method,
          iw_norm=iw_norm,
          natural=natural,
          bound=bound,
          delta=delta,
          seed=seed,
          policy=policy,
          max_offline_iters=max_offline_iters,
          gamma=gamma,
          center_return=center,
          njobs=njobs,
          clipping=clipping,
          entropy=entropy,
          max_iters=max_iters,
          positive_return=positive_return)
