import numpy as np
import warnings
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import colorize
from baselines.common import zipsame
from contextlib import contextmanager
from collections import deque
from baselines import logger
from baselines.common.cg import cg

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize(
        'done in %.3f seconds' % (time.time() - tstart), color='magenta'))


def traj_segment_generator(pi, env, n_episodes, horizon, stochastic=True):
    """
    Generates a dataset of trajectories
        pi: policy
        env: environment
        n_episodes: batch size
        horizon: max episode length
        stochastic: activates policy stochasticity
    """
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()
    cur_ep_ret = 0
    cur_ep_len = 0

    # Initialize history arrays
    ep_rets = []
    ep_lens = []
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    # Collect trajectories
    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        if i == n_episodes:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred*(1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "mask": mask}
            _, vpred = pi.act(stochastic, ob)

            # Reset episode
            ep_rets = []
            ep_lens = []
            mask = np.ones(horizon * n_episodes, 'float32')
            i = 0
            t = 0

        # Update history arrays
        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac
        # Transition
        ob, rew, new, _ = env.step(ac)
        rews[t] = rew
        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1

        # Next episode
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

        # Next step
        t += 1


def add_disc_rew(seg, gamma):
    """
    Adds discounted rewards and returns to the dataset
        seg: the dataset
        gamma: discount factor
    """
    new = np.append(seg['new'], 1)
    rew = seg['rew']
    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)
    seg['ep_disc_rets'] = ep_disc_ret = np.empty(n_ep, 'float32')
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


def line_search_parabola(theta_init, alpha, natural_gradient, set_parameter,
                         evaluate_bound, delta_bound_tol=1e-4,
                         max_line_search_ite=30):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_bound()
    theta_old = theta_init

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * alpha * natural_gradient
        set_parameter(theta)

        bound = evaluate_bound()

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


def optimize_offline(theta, old_thetas_list, set_parameter,
                     evaluate_behav, evaluate_miw, max_offline_ite=1):

    # Compute MISE's denominator
    den_mise = 0
    for i in range(len(old_thetas_list)):
        den_mise += np.exp(
            evaluate_behav([old_thetas_list[i]])).astype(np.float32)

    for i in range(max_offline_ite):
        miw = evaluate_miw(den_mise)
        # parameters update
        # ...theta = ...
        set_parameter(theta)

    return


def render(env, pi, horizon):
    """
    Shows a test episode on the screen
        env: environment
        pi: policy
        horizon: episode length
    """
    t = 0
    ob = env.reset()
    env.render()

    done = False
    while not done and t < horizon:
        ac, _ = pi.act(True, ob)
        ob, _, done, _ = env.step(ac)
        time.sleep(0.1)
        env.render()
        t += 1


def learn(make_env, make_policy, *,
          horizon,
          delta,
          gamma,
          max_iters,
          sampler=None,
          iw_norm='none',
          save_weights=False,
          render_after=None,
          callback=None):
    """
    Learns a policy from scratch
        make_env: environment maker
        make_policy: policy maker
        horizon: max episode length
        delta: probability of failure
        gamma: discount factor
        max_iters: total number of learning iteration
    """

    # Print options
    np.set_printoptions(precision=3)

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space
    max_samples = horizon

    # Build the policy
    pi = make_policy('pi', ob_space, ac_space)
    oldpi = make_policy('oldpi', ob_space, ac_space)

    # Store behaviorals' params and their trajectories
    old_thetas_list = []
    old_traj_list = []

    # Get all learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list
                if v.name.split('/')[1].startswith('pol')]
    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_parameters = sum(shapes)

    # Placeholders
    old_thetas_ = tf.placeholder(shape=[None, n_parameters],
                                 dtype=tf.float32, name='old_thetas')
    ob_ = ob = U.get_placeholder_cached(name='ob')
    ac_ = pi.pdtype.sample_placeholder([max_samples], name='ac')
    mask_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='mask')
    rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='rew')
    disc_rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples),
                               name='disc_rew')
    gradient_ = tf.placeholder(dtype=tf.float32,
                               shape=(n_parameters, 1), name='gradient')
    iter_number_ = tf.placeholder(dtype=tf.int32, name='iter_number')
    den_mise_ = tf.placeholder(dtype=tf.float32, name='den_mise')
    losses_with_name = []

    # Policy densities
    target_log_pdf = pi.pd.logp(ac_)
    behav_log_pdf = oldpi.pd.logp(ac_)

    # Mask operations
    disc_rew_masked = disc_rew_ * mask_
    rew_masked = rew_ * mask_
    target_log_masked = target_log_pdf * mask_
    behav_log_masked = behav_log_pdf * mask_

    # Multiple importance weights computation
    target_sum_log = tf.reduce_sum(target_log_masked)
    behav_sum_log = tf.reduce_sum(behav_log_masked, axis=0)
    log_ratio = target_sum_log - den_mise_
    miw = tf.exp(log_ratio)

    # Renyi divergence
    # ...

    # Return
    ep_return = tf.reduce_sum(disc_rew_masked)

    # Bound definitions
    # ...

    # Baselines' functions
    compute_behav = U.function([ob_, ac_, mask_, old_thetas_],
                               behav_sum_log,
                               updates=None, givens=None)
    compute_miw = U.function([ob_, ac_, mask_, den_mise_], miw,
                             updates=None, givens=None)
    # compute_bound = U.function([ob_, ac_, rew_, disc_rew_, mask_, iter_number_],
    #                            [bound_, assert_ops, print_ops])

    # Info
    losses_with_name.extend([(ep_return, 'Return')])
    losses, loss_names = map(list, zip(*losses_with_name))
    compute_losses = U.function(
        [ob_, ac_, rew_, disc_rew_, mask_, iter_number_], losses)

    # Tf utils
    assert_ops = tf.group(*tf.get_collection('asserts'))
    print_ops = tf.group(*tf.get_collection('prints'))
    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    # Set sampler (default: sequential)
    if sampler is None:
        # Rb:collect only ONE trajectory
        seg_gen = traj_segment_generator(pi, env, 1,
                                         horizon, stochastic=True)
        sampler = type("SequentialSampler", (object,),
                       {"collect": lambda self, _: seg_gen.__next__()})()

    # Tf initialization
    U.initialize()

    # Learning loop
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    while True:
        iters_so_far += 1

        # Render one episode
        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        # Custom callback
        if callback:
            callback(locals(), globals())

        # Exit loop in the end
        if iters_so_far >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iters_so_far)
        # Store the list of arrays representing pi's parameters

        # Generate trajectories
        theta = get_parameter()
        with timed('sampling'):
            seg = sampler.collect(theta)

        # Store the list of arrays representing behaviorals' parameters
        old_thetas_list.append(theta)

        # Retrieve data
        add_disc_rew(seg, gamma)
        lens, u_rets = seg['ep_lens'], seg['ep_rets']
        assert len(lens) == 1
        episodes_so_far += 1
        timesteps_so_far += lens[0]
        ob, ac, rew, disc_rew, mask, iter_number = \
            seg['ob'], seg['ac'], seg['rew'], seg['disc_rew'], seg['mask'], iters_so_far
        print('ARGSSSS:', ob.shape)

        # Info
        with timed('summaries before'):
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("URet", u_rets[0])
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        # Save policy parameters to disk
        # if save_weights:
        #     logger.record_tabular('Weights', str(get_parameter()))
        #     import pickle
        #     file = open('checkpoint.pkl', 'wb')
        #     pickle.dump(theta, file)

        def evaluate_behav(behav_thetas):
            args_behav = ob, ac, mask, behav_thetas
            return compute_behav(*args_behav)

        def evaluate_miw(den_mise):
            args_miw = ob, ac, mask, den_mise
            return compute_miw(*args_miw)

        # Perform optimization
        with timed("Optimization"):
            optimize_offline(theta, old_thetas_list, set_parameter,
                             evaluate_behav, evaluate_miw,
                             max_offline_ite=100)

        # set_parameter(theta)

        # Info
        with timed('summaries after'):
            meanlosses = np.array(compute_losses(*args))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

        # Print all info in a table
        logger.dump_tabular()

    # Close environment in the end
    env.close()
