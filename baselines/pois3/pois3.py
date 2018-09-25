'''
    GPU-friendly implementation of the POIS algorithm, for now only in the
    Control-based setting.
'''
import numpy as np
import warnings
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import zipsame, colorize
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

class Model(object):

    def __init__(self, *, make_policy, ob_space, ac_space, nbatch_act):
        # Get the session
        self.sess = tf.get_default_session()
        # Create policies
        pi = make_policy('pi', self.sess, ob_space, ac_space, nbatch_act)
        oldpi = make_policy('oldpi', self.sess, ob_space, ac_space, nbatch_act)
        # Extracting probabilities from policies
        self.A = pi.pdtype.sample_placeholder([nbatch_act])
        self.logpac = pi.pd.logp(self.A)
        print(self.A)
        #
        self.step = pi.step

    def foo(self, actions):
        return self.sess.run(self.logpac, {self.A: actions})

# Implements a sampler in the vectorized environment
class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.nsteps = nsteps
        # Reset env and save observation
        self.obs = np.zeros((nenv, ) + env.observation_space.shape)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(nenv)]

    def run(self):
        for _ in range(self.nsteps):
            actions, values = self.model.step(self.obs, _stochastic=True)
            self.env.step_async(actions)
            self.obs[:], rewards, self.dones, infos = self.env.step_wait()
            print(self.dones)
            print(self.model.foo(actions))
            exit(0)
        return None

def learn(env, make_policy,
          n_episodes,
          horizon,
          delta,
          gamma,
          max_iters,
          sampler=None,
          use_natural_gradient=False, #can be 'exact', 'approximate'
          fisher_reg=1e-2,
          iw_method='is',
          iw_norm='none',
          bound='J',
          line_search_type='parabola',
          save_weights=False,
          improvement_tol=0.,
          center_return=False,
          render_after=None,
          max_offline_iters=100,
          callback=None):

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda: Model(make_policy=make_policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs)

    model = make_model()
    runner = Runner(env=env, model=model, nsteps=horizon, gamma=gamma)

    U.initialize()

    iters_so_far = 0

    while True:

        iters_so_far += 1

        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        if callback:
            callback(locals(), globals())

        if iters_so_far >= max_iters:
            print('Finised...')
            break

        logger.log('********** Iteration %i ************' % iters_so_far)

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
