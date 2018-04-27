from baselines.common import zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from contextlib import contextmanager
import scipy.stats as sts
import scipy.sparse.linalg as scila
import warnings

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

def add_vtarg_and_adv(seg, gamma):
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

        #print('bound %s - init %s' % (bound, bound_init))
        #print(epsilon)
        #print(delta_bound)
        #print(delta_bound <= delta_bound_opt)

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

        if abs(epsilon_old - epsilon) < 1e-12:
            break

    return theta_opt, epsilon_opt, delta_bound_opt, i_opt+1


def optimize_offline(theta_init, set_parameter, line_search, evaluate_loss, evaluate_gradient, evaluate_natural_gradient=None, gradient_tol=1e-4, bound_tol=1e-4, max_offline_ite=100):
    theta = theta_init
    improvement = 0.
    set_parameter(theta)

    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", "gradient norm", "delta bound ite", "delta bound tot"))

    for i in range(max_offline_ite):
        bound = evaluate_loss()
        gradient = evaluate_gradient()

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

        if np.dot(gradient, natural_gradient) < 0:
            warnings.warn('NatGradient dot Gradient < 0! Using vanulla gradient')
            natural_gradient = gradient

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
          sampler=None,
          use_natural_gradient=False, #can be 'exact', 'approximate'
          fisher_reg=1e-2,
          iw_method='is',
          iw_norm='none',
          bound='student',
          ess_correction=False,
          line_search_type='binary',
          save_weights=False,
          callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    # Setup losses and stuff
    # ----------------------------------------

    if line_search_type == 'binary':
        line_search = line_search_binary
    elif line_search_type == 'parabola':
        line_search = line_search_parabola
    else:
        raise ValueError()

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

    disc_rew_split = tf.stack(tf.split(disc_rew * mask, num_episodes))
    logratio_split = tf.stack(tf.split(logratio * mask, num_episodes))

    target_logpdf_split = tf.split(target_logpdf * mask, num_episodes)
    mask_split = tf.stack(tf.split(mask, num_episodes))

    alpha = 2.
    emp_d = pi.pd.renyi(oldpi.pd, alpha) * mask
    emp_d_split = tf.split(emp_d, num_episodes)
    emp_d_cum_split = tf.reduce_sum(emp_d_split, axis=1)
    empirical_d2 = tf.reduce_mean(tf.exp(emp_d_cum_split))

    ep_return = tf.reduce_sum(mask_split * disc_rew_split, axis=1)

    if iw_method == 'pdis':
        raise NotImplementedError()
    elif iw_method == 'is':
        iw = tf.exp(tf.reduce_sum(logratio_split, axis=1))
        if iw_norm == 'none':
            iwn = iw / num_episodes
            w_return_mean = tf.reduce_sum(iwn * ep_return)
            w_return_std = tf.sqrt(tf.reduce_sum((iw * ep_return - w_return_mean) ** 2) / (num_episodes - 1))
        elif iw_norm == 'sn':
            iwn = iw / tf.reduce_sum(iw)
            w_return_mean = tf.reduce_sum(iwn * ep_return)
            w_return_std = tf.sqrt(tf.reduce_sum(iwn ** 2 * (ep_return - w_return_mean) ** 2) * num_episodes)
        elif iw_norm == 'regression':
            iwn = iw / num_episodes
            mean_iw = tf.reduce_mean(iw)
            beta = tf.reduce_sum((iw - mean_iw) * ep_return * iw) / (tf.reduce_sum((iw - mean_iw) ** 2) + 1e-24)
            w_return_mean = tf.reduce_mean(iw * ep_return - beta * (iw - 1))
            w_return_std = tf.sqrt(tf.reduce_sum((iw * ep_return - beta * iw - w_return_mean + beta) ** 2) / (num_episodes - 2))
        else:
            raise NotImplementedError()
        ess_classic = tf.linalg.norm(iw, 1) ** 2 / tf.linalg.norm(iw, 2) ** 2
        ess_renyi = num_episodes / empirical_d2
    else:
        raise NotImplementedError()

    return_mean = tf.reduce_mean(ep_return)
    return_std = U.reduce_std(ep_return)
    return_max = tf.reduce_max(ep_return)
    return_min = tf.reduce_min(ep_return)

    if bound == 'student':
        bound_ = w_return_mean - sts.t.ppf(1 - delta, num_episodes - 1) / np.sqrt(num_episodes) * w_return_std
    elif bound == 'd2':
        #bound_ = w_return_mean - sts.t.ppf(1 - delta, num_episodes - 1) / np.sqrt(num_episodes) * return_max * tf.sqrt(empirical_d2)
        bound_ = w_return_mean - np.sqrt((1 - delta) / (delta * num_episodes)) * return_max * tf.sqrt(empirical_d2)
    elif bound == 'd2-correct':
        bound_ = w_return_mean - sts.t.ppf(1 - delta, num_episodes - 1) / np.sqrt(num_episodes) * w_return_std * tf.sqrt(empirical_d2)
    elif bound == 'J':
        bound_ = w_return_mean
    else:
        raise NotImplementedError()

    if use_natural_gradient in ['approximate', 'approx', True]:
        p = tf.placeholder(dtype=tf.float32, shape=[None])
        target_logpdf_episode = tf.reduce_sum(target_logpdf_split * mask_split, axis=1)
        grad_logprob = U.flatgrad(tf.stop_gradient(iwn) * target_logpdf_episode, var_list)
        dot_product = tf.reduce_sum(grad_logprob * p)
        hess_logprob = U.flatgrad(dot_product, var_list)
        compute_linear_operator = U.function([p, ob, ac, disc_rew, mask], [-hess_logprob])
    elif use_natural_gradient == 'exact':
        log_grads = tf.stack(
            [U.flatgrad(target_logpdf_split[i] * mask_split[i], var_list) for i in range(len(target_logpdf_split))])
        fisher = tf.matmul(tf.transpose(log_grads), tf.matmul(tf.diag(iwn), log_grads))
        nat_grad = tf.squeeze(tf.linalg.solve(fisher + fisher_reg * np.eye(n_parameters), gradient_))
        compute_natural_gradient = U.function([gradient_, ob, ac, disc_rew, mask], [nat_grad])

    losses = [bound_, return_mean, return_max, return_min, return_std, empirical_d2, w_return_mean, w_return_std, tf.reduce_max(iwn), tf.reduce_min(iwn), tf.reduce_mean(iwn), U.reduce_std(iwn), tf.reduce_max(iw), tf.reduce_min(iw), tf.reduce_mean(iw), U.reduce_std(iw), ess_classic, ess_renyi]
    loss_names = ['Bound', 'InitialReturnMean', 'InitialReturnMax', 'InitialReturnMin', 'InitialReturnStd', 'EmpiricalD2', 'ReturnMeanIW', 'ReturnStdIW', 'MaxIWNorm', 'MinIWNorm', 'MeanIWNorm', 'StdIWNorm', 'MaxIW', 'MinIW', 'MeanIW', 'StdIW', 'ESSClassic', 'ESSRenyi']
    compute_lossandgrad = U.function([ob, ac, disc_rew, mask], losses + [U.flatgrad(bound_, var_list)])
    compute_grad = U.function([ob, ac, disc_rew, mask], [U.flatgrad(bound_, var_list)])
    compute_bound = U.function([ob, ac, disc_rew, mask], [bound_])

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

    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------

    if sampler is None:
        seg_gen = traj_segment_generator(pi, env, num_episodes, horizon, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=num_episodes)
    rewbuffer = deque(maxlen=num_episodes)

    def log_info(args, ite, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far):
        with tf.name_scope("summaries"):
            logger.record_tabular("Itaration", ite)
            meanlosses = np.array(compute_losses(*args))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))

            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if save_weights:
                logger.record_tabular('Weights', str(get_parameter()))

            if rank == 0:
                logger.dump_tabular()
                print(get_parameter())

    improvement_tol = 0.
    theta = get_parameter()

    #fun = U.function([ob, ac, disc_rew, mask], [iw_split, iwn_split,ep_return, return_mean, iwn_split_last])

    fun = U.function([], [pi.ob_rms.mean, pi.ob_rms.std])

    while True:

        iters_so_far += 1

        if callback:
            callback(locals(), globals())

        if iters_so_far >= iters:
            break

        logger.log("********** Iteration %i ************" % iters_so_far)

        with timed("sampling"):
            if sampler is None:
                seg = seg_gen.__next__()
            else:
                seg = sampler.collect(get_parameter())

        add_vtarg_and_adv(seg, gamma)

        lens, rets = seg["ep_lens"], seg["ep_rets"]
        lenbuffer.extend(lens)
        rewbuffer.extend(rets)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        ob, ac, mask, disc_rew = seg["ob"], seg["ac"], seg["mask"], seg["disc_rew"]
        valid_disc_rew = disc_rew[mask.astype(np.bool)]
        disc_rew_standard = np.zeros_like(disc_rew)
        disc_rew_standard[mask.astype(np.bool)] = (valid_disc_rew - np.mean(valid_disc_rew)) / np.std(valid_disc_rew)

        args_standard = ob, ac, disc_rew_standard, mask
        args = ob, ac, disc_rew, mask

        #if hasattr(pi, "ob_rms"):
        #    pi.ob_rms.update(ob[mask.astype(np.bool)])\

        assign_old_eq_new()

        #log_info(args, iters_so_far, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far)

        def evaluate_loss():
            loss = compute_bound(*args_standard)
            return loss[0]

        def evaluate_gradient():
            gradient = compute_grad(*args_standard)
            return gradient[0]

        if use_natural_gradient in ['approximate', 'approx', True]:
            def evaluate_fisher_vector_prod(x):
                return compute_linear_operator(x, *args_standard)[0] + fisher_reg * x

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

        with timed("computegrad"):
            gradient = evaluate_gradient()

        with timed("offline optimization"):

            theta, improvement = optimize_offline(theta,
                                                  set_parameter,
                                                  line_search,
                                                  evaluate_loss,
                                                  evaluate_gradient,
                                                  evaluate_natural_gradient)

            set_parameter(theta)

        #res = fun(*args_standard)
        log_info(args, iters_so_far, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far)

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

    log_info(args, iters_so_far, lenbuffer, rewbuffer, lens, episodes_so_far, timesteps_so_far)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
