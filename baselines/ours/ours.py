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
        if new:
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

def add_vtarg_and_adv(seg, gamma, iw_method='pdis'):
    new = np.append(seg['new'], 1)
    rew = seg['rew']

    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)

    seg['ep_disc_ret'] = ep_disc_ret = np.empty(n_ep, 'float32')
    seg['disc_rew'] = disc_rew = np.empty(n_samp, 'float32')

    #seg['mask1'] = mask1 = np.zeros((n_samp, n_samp), 'float32')
    #seg['mask2'] = mask2 = np.zeros((n_samp, n_ep), 'float32')

    discounter = 0
    ret = 0.
    i = 0
    for t in range(n_samp):
        disc_rew[t] = rew[t] * gamma ** discounter
        ret += disc_rew[t]

        #mask1[t-discounter:t+1, t] = 1

        #rows1 = np.concatenate((rows1, np.arange(t - discounter, t + 1)))
        #cols1 = np.concatenate((cols1, np.repeat([t], discounter)))

        '''
        if iw_method == 'is':
            mask1[t, t - discounter:t + 1] = 1

            rows1 = np.concatenate((rows1, np.repeat([t], discounter)))
            cols1 = np.concatenate((cols1, np.arange(t - discounter, t + 1)))

        mask2[t, i] = 1
        rows2.append(t)
        cols2.append(i)
        '''
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
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i+1
            else:
                return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1

def optimize_offline(theta_init, set_parameter, evaluate_loss, evaluate_gradient, evaluate_natural_gradient=None, gradient_tol=1e-5, bound_tol=1e-3, fisher_reg=1e-10, max_offline_ite=200):
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
        if evaluate_natural_gradient is not None:
            natural_gradient = evaluate_natural_gradient(gradient)
        else:
            natural_gradient = gradient
        #print('Gradient %s' % gradient)
        #print('Natural gradient %s' % natural_gradient)
        #print('Dot %s' % np.dot(natural_gradient, gradient))
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
          num_episodes,
          horizon,
          delta,
          callback=None,
          gamma,
          iters,
          bound_name='student',
         ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    train_writer = tf.summary.FileWriter('log' + '/train')

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)

    train_writer.add_graph(tf.get_default_graph())
    train_writer.flush()

    var_list = pi.get_trainable_variables()

    oldpi = policy_func("oldpi", ob_space, ac_space)
    mask1 = tf.placeholder(dtype=tf.float32, shape=[None,None])
    mask2 = tf.placeholder(dtype=tf.float32, shape=[None,None])

    mask = tf.placeholder(dtype=tf.float32, shape=[None])
    disc_rew = tf.placeholder(dtype=tf.float32, shape=[None])
    ep_len = tf.placeholder(dtype=tf.int32, shape=[None])
    importance_weights = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    target_logpdf = pi.pd.logp(ac)
    behavioral_logpdf = oldpi.pd.logp(ac)
    logratio = target_logpdf - behavioral_logpdf # pnew / pold

    disc_rew_split = tf.split(disc_rew, num_episodes)
    logratio_split = tf.split(logratio * mask, num_episodes) #BE CAREFUL! Already masked
    mask_split = tf.split(mask, num_episodes)

    iw_method = 'pdis'
    if iw_method == 'pdis':
        iw_split = tf.exp(tf.cumsum(logratio_split, axis=1))
    elif iw_method == 'is':
        iw_split = tf.tile(tf.expand_dims(tf.exp(tf.reduce_sum(logratio_split, axis=1)), -1), (1,horizon))
    else:
        raise NotImplementedError()

    ep_return = tf.reduce_sum(iw_split * mask_split * disc_rew_split, axis=1)
    return_mean = tf.reduce_mean(ep_return)
    return_std = reduce_std(ep_return)

    #p = tf.placeholder(dtype=tf.float32, shape=[None])
    #grad_logprob = U.flatgrad(tf.stop_gradient(tf.reshape(iw_split, [horizon*num_episodes,])) * target_logpdf * mask, var_list)
    #dot_product = tf.reduce_sum(grad_logprob * p)
    #hess_logprob = U.flatgrad(dot_product, var_list) / num_episodes

    #compute_linear_operator = U.function([p, ob, ac, disc_rew, mask], [-hess_logprob])

    '''
    grad_return = U.flatgrad(return_mean, var_list)
    dot_product = tf.reduce_sum(grad_return * p)
    hess_return = U.flatgrad(dot_product, var_list)
    compute_linear_operator = U.function([p, ob, ac, disc_rew, mask], [-hess_return])
    '''


    #target_logpdf_split = tf.split(target_logpdf * mask, num_episodes)
    #loggrads = tf.stack([U.flatgrad(target_logpdf_split[i], var_list) for i in range(num_episodes)], axis=0)

    '''
    loggrads = [
        tf.concat(axis=0, values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(v), [U.numel(v)])
            for (v, grad) in zip(var_list, tf.gradients(x, var_list)) ])
         for x in target_logpdf_split]
    
    loghess = [tf.gradients(x, var_list) for x in loggrads]
    fisher = tf.stack(loghess, axis=1)
    '''
    #iw_col = tf.expand_dims(iw_split[:, -1], -1)
    #fisher = 1. / num_episodes * tf.tensordot(iw_col * loggrads, loggrads, axes=[[0], [0]])

    #iw = tf.exp(tf.tensordot(logratio, mask1, axes=1))

    #ep_return = tf.tensordot(iw * disc_rew, mask2, axes=1)
    #return_mean = tf.reduce_mean(ep_return)
    #return_std = reduce_std(ep_return)

    renyi = tf.reduce_mean(tf.exp(pi.pd.renyi(oldpi.pd)))

    if bound_name == 'student':
        bound = return_mean - sts.t.ppf(1 - delta, num_episodes - 1) / np.sqrt(num_episodes) * return_std
    elif bound_name == 'chebyshev':
        bound = return_mean - np.sqrt((1. / delta - 1.) * 1. / num_episodes) * renyi ** env.horizon / 2

    losses = [bound, return_mean, return_std]
    loss_names = ['Bound', 'EpDiscRewMean', 'EpDiscRewStd']

    for (lossname, lossval) in zip(loss_names, losses):
        tf.summary.scalar(lossname, lossval)
    merged = tf.summary.merge_all()
    compute_summary = U.function([ob, ac, disc_rew, mask], [merged])

    #compute_lossandgrad = U.function([ob, ac, disc_rew, mask1, mask2], losses + [U.flatgrad(bound, var_list)])
    gradient_ = U.flatgrad(bound, var_list)
    compute_lossandgrad = U.function([ob, ac, disc_rew, mask], losses + [gradient_])

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    #compute_losses = U.function([ob, ac, disc_rew, mask1, mask2], losses)
    compute_losses = U.function([ob, ac, disc_rew, mask], losses)
    #pdiw_f = U.function([ob, ac, disc_rew, mask], [iw_split, disc_rew_split])
    #compute_natural_gradient = U.function([ob, ac, disc_rew, mask], [hessians])

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

    def evaluate_loss(args):
        loss = allmean(np.array(compute_losses(*args)))
        return loss[0]

    def evaluate_gradient(args):
        *loss, gradient = compute_lossandgrad(*args)
        loss = allmean(np.array(loss))
        gradient = allmean(gradient)
        return gradient, loss[0]

    #def evaluate_natural_gradient(args):
        #natural_gradient = compute_natural_gradient(*args)
        #natural_gradient = allmean(np.array(natural_gradient))
        #return natural_gradient


    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    #TODO change this to work on trajectories
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

            summary = compute_summary(*args)[0]
            train_writer.add_summary(summary, ite)
            train_writer.flush()

            if rank == 0:
                logger.dump_tabular()

    improvement_tol =0.
    theta = get_parameter()

    assign_old_eq_new()

    while True:

        ite += 1
        print(iters)

        if callback: callback(locals(), globals())
        if ite >= iters:
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

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



        with timed("computegrad"):
            gradient, _ = evaluate_gradient(args)

        '''
        def evaluate_fisher_vector_prod(x):
            return allmean(compute_linear_operator(x, *args)[0]) + x

        A = scila.LinearOperator((34, 34), matvec=evaluate_fisher_vector_prod)

        from baselines.common.cg import cg
        def evaluate_natural_gradient(gradient):
            #return cg(evaluate_fisher_vector_prod, gradient, cg_iters=1, verbose=False)
            x = scila.cgs(A, gradient, x0=gradient, maxiter=1)
            return x[0]

        '''

        theta, improvement = optimize_offline(theta, set_parameter, lambda: evaluate_loss(args),
                                                                    lambda: evaluate_gradient(args))
                                                                    #evaluate_natural_gradient)
        set_parameter(theta)

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