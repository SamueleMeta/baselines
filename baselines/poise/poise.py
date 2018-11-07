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
    print(colorize('done in %.3f seconds'%(time.time() - tstart), color='magenta'))

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


def optimize_offline(max_offline_ite=100):
    for i in range(max_offline_ite):

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

    #Print options
    np.set_printoptions(precision=3)

    #Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space
    max_samples = horizon

    #Build the policy
    pi = make_policy('pi', ob_space, ac_space)

    #Get all learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]
    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_parameters = sum(shapes)

    #Placeholders
    ob_ = ob = U.get_placeholder_cached(name='ob')
    ac_ = pi.pdtype.sample_placeholder([max_samples], name='ac')
    mask_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='mask')
    rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='rew')
    disc_rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='disc_rew')
    gradient_ = tf.placeholder(dtype=tf.float32, shape=(n_parameters, 1), name='gradient')
    iter_number_ = tf.placeholder(dtype=tf.int32, name='iter_number')
    losses_with_name = []

    #Episodic return
    ep_return = tf.reduce_sum(mask_ * disc_rew_)

    #Other Tf definitions
    # ...

    #Info
    losses_with_name.extend([(ep_return, 'Return')])
    losses, loss_names = map(list, zip(*losses_with_name))
    compute_losses = U.function([ob_, ac_, rew_, disc_rew_, mask_, iter_number_], losses)

    #Tf utils
    assert_ops = tf.group(*tf.get_collection('asserts'))
    print_ops = tf.group(*tf.get_collection('prints'))
    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    #Set sampler (default: sequential)
    if sampler is None:
        seg_gen = traj_segment_generator(pi, env, 1, horizon, stochastic=True) #collect ONE trajectory
        sampler = type("SequentialSampler", (object,), {"collect": lambda self, _: seg_gen.__next__()})()

    #Tf initialization
    U.initialize()

    #Learning loop
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    while True:
        iters_so_far += 1

        #Render one episode
        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        #Custom callback
        if callback:
            callback(locals(), globals())

        #Exit loop in the end
        if iters_so_far >= max_iters:
            print('Finished...')
            break

        #Learning iteration
        logger.log('********** Iteration %i ************' % iters_so_far)

        #Generate trajectories
        theta = get_parameter()
        with timed('sampling'):
            seg = sampler.collect(theta)

        #Retrieve data
        add_disc_rew(seg, gamma)
        lens, u_rets = seg['ep_lens'], seg['ep_rets']
        assert len(lens)==1
        episodes_so_far += 1
        timesteps_so_far += lens[0]
        args = ob, ac, rew, disc_rew, mask, iter_number = seg['ob'], seg['ac'], seg['rew'], seg['disc_rew'], seg['mask'], iters_so_far

        #Info
        with timed('summaries before'):
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("URet", u_rets[0])
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        #Save policy parameters to disk
        if save_weights:
            logger.record_tabular('Weights', str(get_parameter()))
            import pickle
            file = open('checkpoint.pkl', 'wb')
            pickle.dump(theta, file)

        #Perform optimization
        with timed("Optimization"):
            pass
            #...

        #Info
        with timed('summaries after'):
            meanlosses = np.array(compute_losses(*args))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

        #Print all info in a table
        logger.dump_tabular()

    #Close environment in the end
    env.close()
