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
import warnings
import os
import multiprocessing

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
            new = True
            env.done = True

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
            j = 0
        t += 1

def add_vtarg_and_adv(seg, gamma, iw_method='pdis'):
    new = np.append(seg['new'], 1)
    rew = seg['rew']

    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)

    seg['ep_disc_ret'] = ep_disc_ret = np.empty(n_ep, 'float32')
    seg['disc_rew'] = disc_rew = np.empty(n_samp, 'float32')

    discounter = 0
    ret = 0.
    i = 0
    for t in range(n_samp):
        disc_rew[t] = rew[t] * gamma ** discounter
        ret += disc_rew[t]

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


def line_search_parabola(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss, delta_bound_tol=1e-4, max_line_search_ite=20):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_loss()
    theta_old = theta_init

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * alpha * natural_gradient
        set_parameter(theta)

        bound = evaluate_loss()

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return theta_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i+1
            else:
                return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1


def line_search_binary(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss, delta_bound_tol=1e-4, max_line_search_ite=30):
    low = 0.
    high = None
    bound_init = evaluate_loss()
    delta_bound_old = 0.
    theta_opt = theta_init
    i_opt = 0
    delta_bound_opt = 0.
    epsilon_opt = 0.

    epsilon = 1.


    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * natural_gradient * alpha
        set_parameter(theta)

        bound = evaluate_loss()
        delta_bound = bound - bound_init

        #print('constraint %s' % evaluate_constraint())
        #print('bound %s - init %s' % (bound, bound_init))
        #print(epsilon)
        #print(delta_bound)
        #print(delta_bound <= delta_bound_opt or not evaluate_constraint()[0])

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')

        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            theta_opt = theta
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        epsilon_old = epsilon

        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.

        #print(epsilon)

        if abs(epsilon_old - epsilon) < 1e-6:
            break

    return theta_opt, epsilon_opt, delta_bound_opt, i_opt+1


def optimize_offline(theta_init, set_parameter, line_search, evaluate_loss, evaluate_gradient, evaluate_natural_gradient=None, gradient_tol=1e-4, bound_tol=1e-4, max_offline_ite=30):
    theta = theta_init
    improvement = 0.
    set_parameter(theta)

    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", "gradient norm", "delta bound ite", "delta bound tot"))

    for i in range(max_offline_ite):
        gradient, bound = evaluate_gradient()

        if np.any(np.isnan(gradient)):
            warnings.warn('Got NaN gradient! Stopping!')
            return theta, improvement

        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            return theta, improvement

        if evaluate_natural_gradient is not None:
            natural_gradient = evaluate_natural_gradient(gradient)
        else:
            natural_gradient = gradient

        assert np.dot(gradient, natural_gradient) >= 0

        gradient_norm = np.sqrt(np.dot(gradient, natural_gradient))

        if gradient_norm < gradient_tol:
            print("stopping - gradient norm < gradient_tol")
            print(theta)
            return theta, improvement

        alpha = 1. / gradient_norm ** 2
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_parameter, evaluate_loss)

        #assert not np.any(np.isnan(theta))
        #assert not np.isnan(epsilon)
        #assert not np.isnan(delta_bound)

        set_parameter(theta)

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
          gamma,
          iters,
          use_natural_gradient=False, #can be 'exact', 'approximate'
          fisher_reg=1e-2,
          iw_method='is',
          iw_norm='none',
          bound='student',
          ess_correction=False,
          line_search_type='binary',
          callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    # Setup losses and stuff
    # ----------------------------------------

    if line_search_type == 'binary':
        line_search = line_search_binary
    else:
        line_search = line_search_parabola

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    old_all_var_list = oldpi.get_trainable_variables()
    old_var_list = [v for v in old_all_var_list if v.name.split("/")[1].startswith("pol")]

    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_parameters = sum(shapes)

    mask = tf.placeholder(dtype=tf.float32, shape=[None])
    disc_rew = tf.placeholder(dtype=tf.float32, shape=[None])
    gradient_ = tf.placeholder(dtype=tf.float32, shape=[None, None])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    #Policy functions
    target_logpdf = pi.pd.logp(ac)
    behavioral_logpdf = oldpi.pd.logp(ac)
    logratio = target_logpdf - behavioral_logpdf

    disc_rew_split = tf.stack(tf.split(disc_rew, num_episodes))
    logratio_split = tf.stack(tf.split(logratio * mask, num_episodes))

    target_logpdf_split = tf.split(target_logpdf * mask, num_episodes)
    mask_split = tf.stack(tf.split(mask, num_episodes))

    if iw_method == 'pdis':
        iw_split = tf.exp(tf.cumsum(logratio_split, axis=1))
    elif iw_method == 'is':
        iw_split = tf.expand_dims(tf.exp(tf.reduce_sum(logratio_split * mask_split, axis=1)), -1)
        #tf.tile(tf.expand_dims(tf.exp(tf.reduce_sum(logratio_split, axis=1)), -1), (1, horizon))
    else:
        raise NotImplementedError()

    empirical_d1 = tf.exp(tf.reduce_mean(pi.pd.renyi(oldpi.pd, 1.0001)))
    empirical_d2 = tf.exp(tf.reduce_mean(pi.pd.renyi(oldpi.pd, 2.)))
    empirical_d4 = tf.exp(tf.reduce_mean(pi.pd.renyi(oldpi.pd, 4.)))

    #cumulative_empirical_d2 = tf.reduce_mean(iw_split[:, -1])

    cumulative_empirical_d2 = tf.split(pi.pd.renyi(oldpi.pd, 2.), num_episodes) * mask_split
    cumulative_empirical_d2 = tf.reduce_sum(cumulative_empirical_d2, axis=1)
    cumulative_empirical_d2 = tf.reduce_mean(tf.exp(cumulative_empirical_d2))

    ess_classic = tf.linalg.norm(iw_split, 1) ** 2 / tf.linalg.norm(iw_split, 2) ** 2
    ess_renyi = num_episodes / cumulative_empirical_d2 + 1

    if ess_correction == True or ess_correction == 'classic':
        ess = ess_classic
    elif ess_correction == 'renyi':
        ess = ess_renyi

    if iw_norm == 'sn':
        if iw_method == 'pdis':
            raise NotImplementedError()
        iwn_split = (iw_split + 1e-24) / tf.reduce_sum(iw_split + 1e-24, axis=0)
        #iwn_split = iw_split / tf.reduce_sum(iw_split, axis=0)
        iwn_split = tf.squeeze(iwn_split)
        ep_return = tf.reduce_sum(mask_split * disc_rew_split, axis=1)
        return_mean = tf.reduce_sum(ep_return * iwn_split)
        if ess_correction: #non è corretto
            return_std = tf.sqrt(tf.reduce_sum(iwn_split ** 2 * (ep_return - return_mean) ** 2) * num_episodes / (ess - 1))
        else:
            return_std = tf.sqrt(tf.reduce_sum(iwn_split ** 2 * (ep_return - return_mean) ** 2) * num_episodes)
        third_central_moment = tf.reduce_sum(iwn_split ** 3 * (ep_return - return_mean) ** 3) * num_episodes
        #return_std = tf.sqrt(tf.reduce_sum(iw_split ** 2 * (ep_return - return_mean) ** 2) / (num_episodes - 1))
        #third_central_moment = tf.reduce_sum(iw_split ** 3 * (ep_return - return_mean) ** 3) / (num_episodes - 1)
    else:
        iwn_split = iw_split / num_episodes
        ep_return = tf.reduce_sum(iwn_split * mask_split * disc_rew_split, axis=1)
        return_mean = tf.reduce_sum(ep_return)
        return_std = tf.sqrt(tf.reduce_sum((ep_return - return_mean) ** 2) / (num_episodes - 1))
        third_central_moment = tf.reduce_sum(ep_return - return_mean) ** 3 / (num_episodes - 1)

    if bound == 'student':
        if ess_correction: #anche i gradi di libertà dovrebbero essere considerati
            bound_ = return_mean - sts.t.ppf(1 - delta, num_episodes - 1) / np.sqrt(num_episodes) * return_std
        else:
            bound_ = return_mean - sts.t.ppf(1 - delta, num_episodes - 1) / np.sqrt(num_episodes) * return_std
    elif bound == 'ours':
        bound_ = return_mean - np.sqrt(1. / delta - 1) / np.sqrt(num_episodes) * return_std
    elif bound == 'johnson':
        bound_ = return_mean - np.sqrt(1. / delta - 1) / np.sqrt(num_episodes) * return_std + third_central_moment / (6 * num_episodes * return_std ** 2)



    if use_natural_gradient in ['approximate', 'approx', True]:
        p = tf.placeholder(dtype=tf.float32, shape=[None])
        iw_flat = tf.reshape(iw_split, [horizon * num_episodes])
        grad_logprob = U.flatgrad(tf.stop_gradient(iw_flat) * target_logpdf * mask, var_list)
        dot_product = tf.reduce_sum(grad_logprob * p)
        hess_logprob = U.flatgrad(dot_product, var_list) / num_episodes
        compute_linear_operator = U.function([p, ob, ac, disc_rew, mask], [-hess_logprob])
    elif use_natural_gradient == 'exact':
        log_grads = tf.stack(
            [U.flatgrad(target_logpdf_split[i] * mask_split[i], var_list) for i in range(len(target_logpdf_split))])
        fisher = tf.matmul(tf.transpose(log_grads), tf.matmul(tf.diag(iwn_split[:, -1]), log_grads))
        nat_grad = tf.squeeze(tf.linalg.solve(fisher + fisher_reg * np.eye(n_parameters), gradient_))
        compute_natural_gradient = U.function([gradient_, ob, ac, disc_rew, mask], [nat_grad])

    losses = [bound_, return_mean, return_std, cumulative_empirical_d2, empirical_d1, empirical_d2, empirical_d4, tf.reduce_max(iw_split), tf.reduce_min(iw_split), tf.reduce_mean(iw_split), U.reduce_std(iw_split),  tf.reduce_max(iwn_split), tf.reduce_min(iwn_split), tf.reduce_sum(iwn_split), tf.reduce_mean((iwn_split - 1./num_episodes)**2), ess_classic, ess_renyi]
    loss_names = ['Bound', 'EpDiscRewMean', 'EpDiscRewStd', 'CumEmpD2', 'EmpD1', 'EmpD2', 'EmpD4', 'MaxIW', 'MinIW', 'MeanIW', 'StdIW', 'MaxIWNorm', 'MinIWNorm', 'MeanIWNorm', 'StdIWNorm', 'ESSClassic', 'ESSRenyi']

    compute_lossandgrad = U.function([ob, ac, disc_rew, mask], losses + [U.flatgrad(bound_, var_list)])

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, disc_rew, mask], losses)


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


    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, num_episodes, horizon, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    ite = 0

    def log_info(args, ite, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far):
        with tf.name_scope('summaries'):
            logger.record_tabular("Itaration", ite)
            meanlosses = allmean(np.array(compute_losses(*args)))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))

            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

            if rank == 0:
                logger.dump_tabular()
                print(get_parameter())

    improvement_tol = 0.
    theta = get_parameter()

    assign_old_eq_new()

    while True:

        ite += 1

        if callback: callback(locals(), globals())
        if ite >= iters:
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()

        add_vtarg_and_adv(seg, gamma)
        disc_rew_standard = seg["disc_rew"]  / (max(seg["disc_rew"]) - min(seg["disc_rew"]))
        #disc_rew_standard = (seg["disc_rew"] - np.mean(seg["disc_rew"])) / np.std(seg["disc_rew"])
        #disc_rew_standard = seg['disc_rew']
        args = seg["ob"], seg["ac"], disc_rew_standard, seg['mask']
        args2 = seg["ob"], seg["ac"], seg["disc_rew"], seg['mask']

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        #log_info(args, ite, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far)

        def evaluate_loss():
            loss = allmean(np.array(compute_losses(*args)))
            return loss[0]

        def evaluate_gradient():
            *loss, gradient = compute_lossandgrad(*args)
            loss = allmean(np.array(loss))
            gradient = allmean(gradient)
            return gradient, loss[0]


        if use_natural_gradient in ['approximate', 'approx', True]:
            def evaluate_fisher_vector_prod(x):

                return allmean(compute_linear_operator(x, *args)[0]) + fisher_reg * x
            A = scila.LinearOperator((n_parameters, n_parameters), matvec=evaluate_fisher_vector_prod)

            def evaluate_natural_gradient(gradient):
                x = scila.cg(A, gradient, x0=gradient, maxiter=30)
                return x[0]
        elif use_natural_gradient == 'exact':
            def evaluate_natural_gradient(gradient):
                return compute_natural_gradient(gradient[:, np.newaxis], *args)[0]
        else:
            def evaluate_natural_gradient(gradient):
                return gradient

        meanlosses = allmean(np.array(compute_losses(*args2)))
        for (lossname, lossval) in zip(loss_names[:3], meanlosses[:3]):
            logger.record_tabular("Initial"+lossname, lossval)

        with timed("computegrad"):
            gradient, _ = evaluate_gradient()

        with timed("offline optimization"):

            theta, improvement = optimize_offline(theta,
                                                  set_parameter,
                                                  line_search,
                                                  evaluate_loss,
                                                  evaluate_gradient,
                                                  evaluate_natural_gradient)

            set_parameter(theta)

        log_info(args2, ite, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far)

        assign_old_eq_new()

        assert(improvement >= 0.)

        if improvement < improvement_tol:
            print("Stopping!")
            break

    logger.log("********** Final evaluation ************")

    with timed("sampling"):
        seg = seg_gen.__next__()

    add_vtarg_and_adv(seg, gamma)
    args = seg["ob"], seg["ac"], seg["disc_rew"], seg['mask']

    lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
    lens, rews = map(flatten_lists, zip(*listoflrpairs))
    lenbuffer.extend(lens)
    rewbuffer.extend(rews)

    episodes_so_far += len(lens)
    timesteps_so_far += sum(lens)
    iters_so_far += 1

    log_info(args, ite, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
