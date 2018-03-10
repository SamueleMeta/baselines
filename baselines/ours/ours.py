from baselines.common import Dataset, explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
import scipy.stats as sts
import numpy.linalg as la
import scipy.sparse.linalg as scila

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def traj_segment_generator(pi, env, n_episodes, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if i == n_episodes:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "mask" : mask}
            _, vpred = pi.act(stochastic, ob)            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            mask = np.ones(horizon * n_episodes, 'float32')
            i = 0
            t = 0

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1

        if new or j == horizon:
            j = 0
            new = True

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

            next_t = (i+1) * horizon

            mask[t+1:next_t] = 0.
            acs[t+1:next_t] = acs[t]
            obs[t+1:next_t] = obs[t]

            t = next_t - 1
            i += 1
        t += 1

def update_epsilon(delta_bound, epsilon_old, max_increase=2.):
    if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
        return epsilon_old * max_increase
    else:
        return epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))


def line_search(theta_init,
                alpha,
                natural_gradient,
                set_param,
                evaluate_loss,
                delta_bound_tol=1e-2,
                max_line_search_ite=100):

    epsilon = 1. / np.sqrt(alpha)
    delta_bound_old = -np.inf
    bound_init = evaluate_loss()
    theta_old = theta_init

    for i in range(max_line_search_ite):
        theta = theta_init + epsilon * alpha * natural_gradient
        set_param(theta)
        bound = evaluate_loss()
        #print("bound %s - step %s" % (bound, epsilon * alpha))
        #print(theta)
        #print(natural_gradient)
        delta_bound = bound - bound_init

        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if np.isnan(delta_bound) or delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i+1
            else:
                return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1

def optimize_offline(eval_param,
                     set_param,
                     eval_bound,
                     eval_grad,
                     eval_fisher=None,
                     gradient_tol=1e-5,
                     bound_tol=1e-3,
                     fisher_reg=1e-10,
                     max_offline_ite=200):

    theta_init = eval_param()
    theta = theta_init
    improvement = 0.

    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", "gradient norm", "delta bound ite", "delta bound tot"))

    for i in range(max_offline_ite):

        gradient = eval_grad()
        assert not any(np.isnan(gradient))

        if eval_fisher is not None:
            fisher = eval_fisher()
            natural_gradient = la.solve(fisher + fisher_reg * np.eye(fisher.shape[0]), gradient)
        else:
            natural_gradient = gradient

        assert np.dot(gradient, natural_gradient) >= 0

        gradient_norm = np.sqrt(np.dot(gradient, natural_gradient))

        if gradient_norm < gradient_tol:
            print("stopping - gradient norm < gradient_tol")
            return theta, improvement

        alpha = 1. / gradient_norm ** 2
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_param, eval_bound)
        set_param(theta)

        improvement += delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, gradient_norm, delta_bound, improvement))

        if delta_bound < bound_tol:
            print("stopping - delta bound < bound_tol")
            print(theta)
            return theta, improvement

    return theta, improvement

def learn(env, policy_func, *,
          num_episodes,
          horizon,
          delta,
          callback=None,
          gamma,
          iters,
          bound_name='student',
          natural_gradient=True):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    U.initialize()

    seg_gen = traj_segment_generator(pi, env, num_episodes, horizon, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    improvement_tol = 0.
    theta = pi.eval_param()

    assign_old_eq_new()

    eval_param = lambda: pi.eval_param()
    set_param = lambda x: pi.set_param(x)

    while True:

        iters_so_far += 1

        if callback: callback(locals(), globals())
        if iters_so_far >= iters:
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()

        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ob"], seg["ac"], seg["rew"])
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
        lens, rews, states, actions, rewards = map(flatten_lists, zip(*listoflrpairs))

        J = pi.eval_J(states, actions, rewards, lens, behavioral=oldpi, per_decision=True, gamma=gamma)
        var_J = pi.eval_var_J(states, actions, rewards, lens, behavioral=oldpi, per_decision=True, gamma=gamma)

        def eval_bound():
            return pi.eval_bound(states, actions, rewards, lens, behavioral=oldpi, per_decision=True, gamma=gamma, delta=delta)

        def eval_grad():
            return pi.eval_grad_J(states, actions, rewards, lens, behavioral=oldpi, per_decision=True, gamma=gamma)

        if natural_gradient:
            def eval_fisher():
                return pi.eval_fisher(states, actions, lens, behavioral=oldpi)
        else:
            eval_fisher = None

        theta, improvement = optimize_offline(eval_param, set_param, eval_bound, eval_grad, eval_fisher)
        set_param(theta)
        assign_old_eq_new()

        assert(improvement >= 0.)

        logger.record_tabular("EpBound", eval_bound())
        logger.record_tabular("EpRewMean", np.mean(rews))
        logger.record_tabular("EpDiscRewMean", J)
        logger.record_tabular("EpDiscRewStd", np.sqrt(var_J / num_episodes))
        logger.record_tabular("EpLenMean", np.mean(lens))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

        if improvement < improvement_tol:
            print("Stopping!")
            break

    logger.log("********** Final evaluation ************")




def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]