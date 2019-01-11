# Implementing UCB
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import colorize
from contextlib import contextmanager
from baselines import logger
from collections import defaultdict


def eval_trajectory(env, pol, gamma, horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t < horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        # ob = np.reshape(ob, newshape=s.shape)
        ret += r
        disc_ret += gamma**t * r
        t += 1

    return ret, disc_ret, t


def ucb1(make_env,
         make_policy,
         max_iters,
         gamma,
         horizon,
         feature_fun=None):

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Build the higher level target and behavioral policies
    pi = make_policy('pi', ob_space, ac_space)

    # Get all pi's learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = \
        [v for v in all_var_list if v.name.split('/')[1].startswith('higher')]

    # TF functions
    set_parameters = U.SetFromFlat(var_list)
    get_parameters = U.GetFlat(var_list)

    # My Placeholders
    ret_ = tf.placeholder(dtype=tf.float32, shape=(max_iters), name='ret')
    disc_ret_ = tf.placeholder(dtype=tf.float32, shape=(max_iters),
                               name='disc_ret')
    n_ = tf.placeholder(dtype=tf.float32, name='iter_number')
    n_int = tf.cast(n_, dtype=tf.int32)
    mask_iters_ = tf.placeholder(dtype=tf.float32, shape=(max_iters),
                                 name='mask_iters')

    # Calculate the grid of parameters to evaluate
    gain_grid = np.linspace(-1, 1, grid_size)
    logstd_grid = np.linspace(-4, 0, grid_size)
    std_too = (len(rho_init) == 2)
    if std_too:
        threeDplot = True
        x, y = np.meshgrid(gain_grid, logstd_grid)
        X = x.reshape((np.prod(x.shape),))
        Y = y.reshape((np.prod(y.shape),))
        rho_grid = list(zip(X, Y))
    else:
        rho_grid = [[x] for x in gain_grid]

    n_selections = np.zeros(len(rho_grid))
    ret_sums = np.zeros(len(rho_grid))
    for rho in rho_grid:
        n_selections[str(rho)] = 0
        ret_sums[str(rho)] = 0
    regret = 0
    iter = 0
    while True:
        iter += 1

        # Exit loop in the end
        if iter >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iter)

        ub = []
        ub_best = 0
        i_best = 0
        average_ret = []
        bonus = []
        for i, rho in enumerate(rho_grid):
            if n_selections[i] > 0:
                average_ret_rho = ret_sums[i] / n_selections[i]
                bonus_rho = np.sqrt(2 * np.log(iter) / n_selections[i])
                ub_rho = average_ret + bonus
                ub.append(ub_rho)
            if not std_too:
                average_ret.append(average_ret_rho)
                bonus.append(bonus_rho)
            else:
                ub_rho = 1e100
                ub.append(ub_rho)
            if ub_rho > ub_best:
                ub_best = ub_rho
                rho_best = rho
                i_best = i
        # Sample actor's parameters from chosen arm
        set_parameters(rho_best)
        theta = pi.resample()
        # Store parameters of both the hyperpolicy and the actor
        if env.spec.id == 'LQG1D-v0':
            mu1_actor = pi.eval_actor_mean([[1]])[0][0]
            mu1_higher = pi.eval_higher_mean([[1]])[0]
            sigma = pi.eval_higher_std()[0]
            logger.record_tabular("LQGmu1_actor", mu1_actor)
            logger.record_tabular("LQGmu1_higher", mu1_higher)
            logger.record_tabular("LQGsigma_higher", sigma)
        # Sample a trajectory with the newly parametrized actor
        _, disc_ret, _ = eval_trajectory(
            env, pi, gamma, horizon, feature_fun)
        ret_sums[i_best] += disc_ret
        regret += (17.35 - disc_ret)
        logger.record_tabular("Regret", regret)
        n_selections[i_best] += 1
