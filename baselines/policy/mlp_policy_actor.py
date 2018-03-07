from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np


class MlpPolicy(object):
    """Gaussian policy with critic, based on multi-layer perceptron"""
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            U.initialize()

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers,
              gaussian_fixed_var=True, use_bias=True):
        """Params:
            ob_space: task observation space
            ac_space : task action space
            hid_size: width of hidden layers
            num_hid_layers: depth
            gaussian_fixed_var: True->separate parameter, False->two heads
            use_bias: whether to include bias in neurons
        """
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # Actor
        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                                                      name='fc%i' % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0), use_bias=use_bias))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2,
                                       name='final',
                                       kernel_initializer=U.normc_initializer(0.01),
                                       use_bias=use_bias)
                logstd = tf.get_variable(name="pol_logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0],
                                          name='final',
                                          kernel_initializer=U.normc_initializer(0.01))

        # Acting
        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], ac)

        # Evaluating
        self.ob = ob
        self.ac_in = U.get_placeholder(name="ac_in", dtype=tf.float32,
                                       shape=[sequence_length] +
                                             list(ac_space.shape))
        self.gamma = U.get_placeholder(name="gamma",
                                       dtype=tf.float32, shape=[])
        self.rew = U.get_placeholder(name="rew", dtype=tf.float32,
                                     shape=[sequence_length] + [1])
        self.logprobs = self.pd.logp(self.ac_in)

    # Acting
    def act(self, stochastic, ob):
        ac1 = self._act(stochastic, ob[None])
        return ac1[0]

    # Performance evaluation
    def get_performance(self, lens=1, behavioral=None, per_decision=False, gamma=.99):
        assert sum(lens) > 0

        weighted_rets = self._get_weighted_returns(lens, behavioral, per_decision, gamma)
        mean, var = tf.nn.moments(weighted_rets, axes=[0])
        return mean, var

    def eval_performance(self, states, actions, rewards, lens=1, behavioral=None, per_decision=False, gamma=.99,
                         get_var=False):
        assert len(states) == len(actions) == len(rewards)
        assert len(rewards) > 0

        _states, _actions, _rewards = self._prepare(states, actions, rewards, lens)
        J_hat, J_var = self.get_performance(lens, behavioral, per_decision, gamma)
        if behavioral is not None:
            fun = U.function([self.ob, behavioral.ob, self.ac_in, self.rew, self.gamma], [J_hat, J_var])
            return fun(_states, _states, _actions, _rewards, gamma) if get_var else \
            fun(_states, _states, _actions, _rewards, gamma)[0]
        else:
            fun = U.function([self.ob, self.ac_in, self.rew, self.gamma], [J_hat, J_var])
            return fun(_states, _actions, _rewards, gamma) if get_var else fun(_states, _actions, _rewards, gamma)[0]

    def _prepare(self, states, actions, rewards, lens):
        if type(lens) is not list:
            assert len(rewards) % lens == 0
            no_of_samples = len(rewards)
        else:
            no_of_samples = sum(lens)

        _states = np.array(states[:no_of_samples])
        _actions = np.array(actions[:no_of_samples])
        _rewards = np.array(rewards[:no_of_samples])

        if _states.ndim == 1:
            _states = np.expand_dims(_states, axis=1)
        if _actions.ndim == 1:
            _actions = np.expand_dims(_actions, axis=1)
        if _rewards.ndim == 1:
            _rewards = np.expand_dims(_rewards, axis=1)

        return _states, _actions, _rewards

    def _get_weighted_returns(self, lens, behavioral, per_decision, gamma):
        rews_by_episode = tf.split(self.rew, lens)
        rews_by_episode = tf.stack(self._fill(rews_by_episode))

        disc = self.gamma + 0 * rews_by_episode
        disc = tf.cumprod(disc, axis=1, exclusive=True)
        disc_rews = rews_by_episode * disc

        if behavioral is None:
            weighted_rets = tf.reduce_sum(disc_rews, axis=1)
        else:
            log_ratios = self.logprobs - behavioral.pd.logp(self.ac_in)
            log_ratios = tf.expand_dims(log_ratios, axis=1)
            log_ratios_by_episode = tf.split(log_ratios, lens)
            log_ratios_by_episode = tf.stack(self._fill(log_ratios_by_episode))

            if per_decision:
                iw = tf.exp(tf.cumsum(log_ratios_by_episode, axis=1))
                iw = tf.expand_dims(iw, axis=2)
                weighted_rets = tf.reduce_sum(tf.multiply(disc_rews, iw), axis=1)
            else:
                iw = tf.exp(tf.reduce_sum(log_ratios_by_episode, axis=1))
                rets = tf.reduce_sum(disc_rews, axis=1)
                weighted_rets = tf.multiply(rets, iw)

        return weighted_rets

    def _fill(self, tensors, filler=0):
        max_len = max(t.shape[0].value for t in tensors)
        result = []
        for t in tensors:
            padding = filler * tf.ones([max_len - t.shape[0].value] + t.shape[1:].as_list())

            padding = tf.stack([filler + 0 * t[0] for _ in range(max_len - \
                                                                 t.shape[0].value)],
                               axis=0)
            if len(padding.shape) == 1:
                padding = tf.expand_dims(padding, axis=1)
            t = tf.concat([t, padding], axis=0)
            result.append(t)
        return result

    # Weight manipulation
    def eval_param(self):
        """"Policy parameters (numeric,flat)"""

        with tf.variable_scope(self.scope + '/pol') as vs:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
        return U.GetFlat(var_list)()

    def get_param(self):
        """Policy parameters (symbolic, nested)"""

        with tf.variable_scope(self.scope + '/pol') as vs:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                     scope=vs.name)

    def set_param(self, param):
        """Set policy parameters to (flat) param"""

        with tf.variable_scope(self.scope + '/pol') as vs:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
            U.SetFromFlat(var_list)(param)

    # Not used
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []