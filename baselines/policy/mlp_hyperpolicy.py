import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import DiagGaussianPdType, CholeskyGaussianPdType
import numpy as np
from baselines.common import set_global_seeds
import scipy.stats as sts

"""References
PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration
for control." International Conference on Artificial Neural Networks. Springer,
Berlin, Heidelberg, 2008.
"""


class PeMlpPolicy(object):
    """Multi-layer-perceptron policy with
    Gaussian parameter-based exploration"""

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            U.initialize()
            self.scope = tf.get_variable_scope().name
            # Sample initial actor params
            tf.get_default_session().run(self._use_sampled_actor_params)

    def _init(self, ob_space, ac_space, hid_layers=[],
              deterministic=True, diagonal=True,
              use_bias=True, use_critic=False,
              seed=None, verbose=True,
              hidden_W_init=U.normc_initializer(1.0)):
        """Params:
            ob_space: task observation space
            ac_space : task action space
            hid__layers: list with width of each hidden layer
            deterministic: whether the actor is deterministic
            diagonal: whether the higher order policy has a diagonal covariance
            matrix
            use_bias: whether to include bias in neurons
            use_critic: whether to include a critic network
            seed: optional random seed
        """
        # Check environment's shapes
        assert isinstance(ob_space, gym.spaces.Box)
        assert len(ac_space.shape) == 1
        # Set seed
        if seed is not None:
            set_global_seeds(seed)
        # Set some attributes
        self.diagonal = diagonal
        self.use_bias = use_bias
        batch_length = None  # Accepts a sequence of eps of arbitrary length
        self.ac_dim = ac_space.shape[0]
        self.ob_dim = ob_space.shape[0]
        self.linear = not hid_layers
        self.verbose = verbose
        self._ob = ob = U.get_placeholder(
            name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))

        # Actor (N.B.: weight initialization is irrelevant)
        with tf.variable_scope('actor'):
            last_out = ob
            for i, hid_size in enumerate(hid_layers):
                # Mlp feature extraction
                last_out = tf.nn.tanh(
                    tf.layers.dense(last_out, hid_size,
                                    name='fc%i' % (i+1),
                                    kernel_initializer=hidden_W_init,
                                    use_bias=use_bias))
            if deterministic and isinstance(ac_space, gym.spaces.Box):
                # Determinisitc action selection
                self.actor_mean = actor_mean = \
                    tf.layers.dense(last_out, ac_space.shape[0],
                                    name='action',
                                    kernel_initializer=hidden_W_init,
                                    use_bias=use_bias)
            else:
                raise NotImplementedError

        # Get actor flatten weights
        with tf.variable_scope('actor') as scope:
            self.actor_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope.name)
            # flatten weights
            self.flat_actor_weights = tf.concat(
                [tf.reshape(w, [-1]) for w in self.actor_weights], axis=0)
            self._n_actor_weights = n_actor_weights = \
                self.flat_actor_weights.shape[0]

        # Higher order policy (Gaussian)
        with tf.variable_scope('higher'):
            # Initial means sampled from a normal distribution N(0,1)
            higher_mean_init = tf.where(
                tf.not_equal(self.flat_actor_weights, tf.constant(0, dtype=tf.float32)),
                tf.random_normal(shape=[n_actor_weights.value], stddev=0.01),
                tf.zeros(shape=[n_actor_weights]))  # bias init must stay zero
            self.higher_mean = higher_mean = tf.get_variable(
                name='higher_mean', initializer=higher_mean_init)

            if diagonal:
                # Diagonal covariance matrix; all stds initialized to 0
                self.higher_logstd = higher_logstd = \
                    tf.get_variable(name='higher_logstd',
                                    shape=[n_actor_weights],
                                    initializer=tf.initializers.constant(0.))
                pdparam = tf.concat([higher_mean,
                                     higher_mean * 0. + higher_logstd],
                                    axis=0)
                self.pdtype = pdtype = \
                    DiagGaussianPdType(n_actor_weights.value)
            else:
                # Cholesky covariance matrix
                self.higher_logstd = higher_logstd = tf.get_variable(
                    name='higher_logstd',
                    shape=[n_actor_weights*(n_actor_weights + 1)//2],
                    initializer=tf.initializers.constant(0.))
                pdparam = tf.concat([higher_mean, higher_logstd],
                                    axis=0)
                self.pdtype = pdtype = CholeskyGaussianPdType(
                    n_actor_weights.value)

        # Sample actor weights
        self.pd = pdtype.pdfromflat(pdparam)
        sampled_actor_params = self.pd.sample()
        symm_sampled_actor_params = self.pd.sample_symmetric()
        self._sample_actor_params = U.function([], [sampled_actor_params])
        self._sample_symm_actor_params = U.function(
            [], list(symm_sampled_actor_params))

        # Assign actor weights
        with tf.variable_scope('actor') as scope:
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=scope.name)
            self._use_sampled_actor_params = \
                U.assignFromFlat(actor_params, sampled_actor_params)
            self._get_actor_params = U.GetFlat(actor_params)
            self._set_actor_params = U.SetFromFlat(actor_params)

        # Act
        self._action = action = actor_mean
        self._act = U.function([ob], [action])

        # Manage higher policy weights
        with tf.variable_scope('higher') as scope:
            self._higher_params = higher_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            self.flat_higher_params = tf.concat([tf.reshape(w, [-1]) for w in
                                                 self._higher_params], axis=0)
            self._n_higher_params = self.flat_higher_params.shape[0]
            self._get_flat_higher_params = U.GetFlat(higher_params)
            self._set_higher_params = U.SetFromFlat(self._higher_params)

        # Evaluating
        self._actor_params_in = actor_params_in = \
            U.get_placeholder(name='actor_params_in',
                              dtype=tf.float32,
                              shape=[batch_length] + [n_actor_weights])
        self._rets_in = rets_in = \
            U.get_placeholder(name='returns_in',
                              dtype=tf.float32,
                              shape=[batch_length])
        ret_mean, ret_std = tf.nn.moments(rets_in, axes=[0])
        self._get_ret_mean = U.function([self._rets_in], [ret_mean])
        self._get_ret_std = U.function([self._rets_in], [ret_std])
        self._logprobs = logprobs = self.pd.logp(actor_params_in)
        pgpe_times_n = U.flatgrad(logprobs*rets_in, higher_params)
        self._get_pgpe_times_n = U.function([actor_params_in, rets_in],
                                            [pgpe_times_n])
        self._get_actor_mean = U.function([ob], [self.actor_mean])

        # Batch off-policy PGPE
        self._probs = tf.exp(logprobs)
        self._behavioral = None
        self._renyi_other = None

        # Renyi computation
        self._det_sigma = tf.exp(tf.reduce_sum(self.higher_logstd))

        # Fisher computation (diagonal case)
        mean_fisher_diag = tf.exp(-2*self.higher_logstd)
        cov_fisher_diag = mean_fisher_diag*0 + 2
        self._fisher_diag = tf.concat(
            [mean_fisher_diag, cov_fisher_diag], axis=0)
        self._get_fisher_diag = U.function([], [self._fisher_diag])

    # Black box usage
    def act(self, ob, resample=False):
        """
        Sample weights for the actor network, then sample action(s) from the
        resulting actor depending on state(s)

        Params:
               ob: current state, or a list of states
               resample: whether to resample actor params before acting
        """

        if resample:
            actor_param = self.resample()

        action = self._act(np.atleast_2d(ob))[0]
        return (action, actor_param) if resample else action

    def resample(self):
        """Resample actor params

        Returns:
            the sampled actor params
        """
        tf.get_default_session().run(self._use_sampled_actor_params)
        return self.eval_actor_params()

    def act_with(self, ob, actor_params):
        self.set_actor_params(actor_params)
        return self.act(ob)

    def seed(self, seed):
        if seed is not None:
            set_global_seeds(seed)

    def eval_mean(self, ob):
        return self._get_actor_mean(ob)[0]

    # Weights manipulation
    def set_params(self, new_higher_params):
        """Set higher order policy parameters from flat sequence"""
        self._set_higher_params(new_higher_params)

    def eval_params(self):
        """Get current params of the higher order policy"""
        return self._get_flat_higher_params()

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    # Direct actor policy manipulation
    def draw_actor_params(self):
        """Sample params for an actor (without using them)"""
        sampled_actor_params = self._sample_actor_params()[0]
        return sampled_actor_params

    def draw_symmetric_actor_params(self):
        return tuple(self._sample_symm_actor_params())

    def eval_actor_params(self):
        """Get actor params as last assigned"""
        return self._get_actor_params()

    def set_actor_params(self, new_actor_params):
        """Manually set actor policy parameters from flat sequence"""
        self._set_actor_params(new_actor_params)

    # Distribution properties
    def eval_renyi(self, other, order=2):
        """Renyi divergence
            Special case: order=1 is kl divergence

        Params:
            other: policy to evaluate the distance from
            order: order of the Renyi divergence
            exponentiate: if true, actually returns e^Renyi(self||other)
        """
        if other is not self._renyi_other:
            if self.verbose:
                print('Building graph')
            self._renyi_order = tf.placeholder(name='renyi_order',
                                               dtype=tf.float32,
                                               shape=[])
            self._renyi_other = other
            if order < 1:
                raise ValueError('Order must be >= 1')
            else:
                renyi = self.pd.renyi(other.pd, alpha=self._renyi_order)
                self._get_renyi = U.function([self._renyi_order], [renyi])

        return self._get_renyi(order)[0]

    def eval_fisher(self):
        if not self.diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return np.ravel(self._get_fisher_diag()[0])

    def fisher_product(self, x):
        if not self.diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return x/self.eval_fisher()
