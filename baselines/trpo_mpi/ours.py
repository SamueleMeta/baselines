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
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

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

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, iw_method='pdis'):
    new = np.append(seg['new'], 1)
    rew = seg['rew']

    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)

    seg['ep_disc_ret'] = ep_disc_ret = np.empty(n_ep, 'float32')
    seg['disc_rew'] = disc_rew = np.empty(n_samp, 'float32')

    rows1, cols1 = [], []
    rows2, cols2 = [], []


    seg['mask1'] = mask1 = np.zeros((n_samp, n_samp), 'float32')
    seg['mask2'] = mask2 = np.zeros((n_samp, n_ep), 'float32')

    discounter = 0
    ret = 0.
    i = 0
    for t in range(n_samp):
        disc_rew[t] = rew[t] * gamma ** discounter
        ret += disc_rew[t]

        mask1[t-discounter:t+1, t] = 1

        rows1 = np.concatenate((rows1, np.arange(t - discounter, t + 1)))
        cols1 = np.concatenate((cols1, np.repeat([t], discounter)))

        if iw_method == 'is':
            mask1[t, t - discounter:t + 1] = 1

            rows1 = np.concatenate((rows1, np.repeat([t], discounter)))
            cols1 = np.concatenate((cols1, np.arange(t - discounter, t + 1)))

        mask2[t, i] = 1
        rows2.append(t)
        cols2.append(i)

        if new[t + 1]:
            discounter = 0
            ep_disc_ret[i] = ret
            i += 1
            ret = 0.
        else:
            discounter += 1

def update_epsilon(delta_bound, epsilon_old, max_increase=2.):
    if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
        return epsilon_old * max_increase
    else:
        return epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))


def line_search(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss, delta_bound_tol=1e-2, max_line_search_ite=100):
    epsilon = 1. / np.sqrt(alpha)
    delta_bound_old = -np.inf
    bound_init = evaluate_loss()
    theta_old = theta_init

    for i in range(max_line_search_ite):
        theta = theta_init + epsilon * alpha * natural_gradient
        set_parameter(theta)
        bound = evaluate_loss()
        delta_bound = bound - bound_init

        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if delta_bound <= delta_bound_old + delta_bound_tol:
            return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1

def optimize_offline(theta_init, set_parameter, evaluate_loss, evaluate_gradient, gradient_tol=1e-5, bound_tol=1e-3, fisher_reg=1e-10, max_offline_ite=100):
    theta = theta_init
    improvement = 0.
    set_parameter(theta)

    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", "gradient norm", "delta bound ite", "delta bound tot"))

    for i in range(max_offline_ite):
        gradient, bound = evaluate_gradient()
        #TODO compute fisher
        #fisher = 0.
        #natural_gradient = la.solve(fisher + fisher_reg * np.eye(fisher.shape[0]), gradient)
        natural_gradient = gradient
        gradient_norm = np.sqrt(np.dot(gradient, natural_gradient))
        if gradient_norm < gradient_tol:
            print("stopping - gradient norm < gradient_tol")
            return theta, improvement

        alpha = 1. / gradient_norm ** 2
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_parameter, evaluate_loss)
        set_parameter(theta)

        improvement += delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, gradient_norm, delta_bound, improvement))

        if delta_bound < bound_tol:
            print("stopping - delta bound < bound_tol")
            print(theta)
            return theta, improvement

    return theta, improvement

def learn(env, policy_func, *,
        timesteps_per_batch, # what to train on
        delta,
        N,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        gamma,
          iters,
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)

    var_list = pi.get_trainable_variables()

    oldpi = policy_func("oldpi", ob_space, ac_space)
    mask1 = tf.placeholder(dtype=tf.float32, shape=[None,None])
    mask2 = tf.placeholder(dtype=tf.float32, shape=[None,None])
    disc_rew = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    logratio = pi.pd.logp(ac) - oldpi.pd.logp(ac) # pnew / pold
    iw = tf.exp(tf.tensordot(logratio, mask1, axes=1))

    ep_return = tf.tensordot(iw * disc_rew, mask2, axes=1)
    return_mean = tf.reduce_mean(ep_return)
    return_std = reduce_std(ep_return)

    bound = return_mean - sts.t.ppf(1 - delta, N - 1) / np.sqrt(N) * return_std

    losses = [bound, return_mean, return_std]
    loss_names = ['bound', 'return', 'std']

    compute_lossandgrad = U.function([ob, ac, disc_rew, mask1, mask2], losses + [U.flatgrad(bound, var_list)])


    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, disc_rew, mask1, mask2], losses)

    pdiw_f = U.function([ob, ac, mask1, mask2, disc_rew], [iw, ep_return, var_list])

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    def evaluate_loss():
        loss = allmean(np.array(compute_losses(*args)))
        return loss[0]

    def evaluate_gradient():
        *loss, gradient = compute_lossandgrad(*args)
        loss = allmean(np.array(loss))
        gradient = allmean(gradient)
        return gradient, loss[0]

    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1
    schedule = 'constant'
    optim_stepsize = 1e-2
    ite = 0

    improvement_tol = 1e-2
    theta = get_parameter()

    while True:
        ite+=1
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma)

        ob, ac, disc_rew, mask1, mask2 = seg["ob"], seg["ac"], seg["disc_rew"], seg['mask1'], seg['mask2']

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], disc_rew, mask1, mask2
        assign_old_eq_new() # set old parameter values to new parameter values

        print(theta)

        with timed("computegrad"):
            gradient = evaluate_gradient()
        print(gradient)

        theta, improvement = optimize_offline(theta, set_parameter, evaluate_loss, evaluate_gradient)
        set_parameter(theta)

        if improvement < improvement_tol:
            break

        meanlosses = bound, _, _ = allmean(np.array(compute_losses(*args)))
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]