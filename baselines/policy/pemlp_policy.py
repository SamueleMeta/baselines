from baselines.common.running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import DiagGaussianPdType, CholeskyGaussianPdType
import numpy as np
from baselines.common import set_global_seeds
import scipy.stats as sts

"""References
PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
control." International Conference on Artificial Neural Networks. Springer,
Berlin, Heidelberg, 2008.
"""

class PeMlpPolicy(object):
    """Multi-layer-perceptron policy with Gaussian parameter-based exploration"""
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        #with tf.device('/cpu:0'):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            U.initialize()
            #Sample initial actor params
            tf.get_default_session().run(self._use_sampled_actor_params)

    def _init(self, ob_space, ac_space, hid_layers=[],
              deterministic=True, diagonal=True,
              use_bias=True, standardize_input=True, use_critic=False, 
              seed=None):
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
        assert isinstance(ob_space, gym.spaces.Box)
        assert len(ac_space.shape)==1
        self.diagonal = diagonal
        batch_length = None #Accepts a sequence of episodes of arbitrary length

        if seed is not None:
            set_global_seeds(seed)

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        #Critic (normally not used)
        if use_critic:
            with tf.variable_scope('critic'):
                obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                last_out = obz if standardize_input else ob
                for i, hid_size in enumerate(hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        #Actor (N.B.: weight initialization is irrelevant)
        with tf.variable_scope('actor'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / np.sqrt(self.ob_rms.var), -5.0, 5.0)
            last_out = obz if standardize_input else ob
            for i, hid_size in enumerate(hid_layers):
                #Mlp feature extraction
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                                                      name='fc%i'%(i+1),
                                                      kernel_initializer=U.normc_initializer(1),use_bias=use_bias))
            if deterministic and isinstance(ac_space, gym.spaces.Box):
                #Determinisitc action selection
                self.actor_mean = actor_mean = tf.layers.dense(last_out, ac_space.shape[0],
                                       name='final',
                                       kernel_initializer=U.normc_initializer(0.01),
                                       use_bias=use_bias)
            else: 
                raise NotImplementedError #Currently supports only deterministic action policies

        #Higher order policy (Gaussian)
        with tf.variable_scope('actor') as scope:
            self.actor_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)
            self.flat_actor_weights = tf.concat([tf.reshape(w, [-1]) for w in \
                                            self.actor_weights], axis=0) #flatten
            self._n_actor_weights = n_actor_weights = self.flat_actor_weights.shape[0]

        with tf.variable_scope('higher'):
            #Initial means sampled from a normal distribution N(0,1)
            self.higher_mean = higher_mean = tf.get_variable(name='higher_mean',
                                               shape=[n_actor_weights],
                                               initializer=tf.initializers.random_normal(stddev=1.))
            if diagonal:
                #Diagonal covariance matrix; all stds initialized to 0
                self.higher_logstd = higher_logstd = tf.get_variable(name='higher_logstd',
                                               shape=[n_actor_weights],
                                               initializer=tf.initializers.constant(0.))
                pdparam = tf.concat([higher_mean, higher_mean * 0. + 
                                   higher_logstd], axis=0)
                self.pdtype = pdtype = DiagGaussianPdType(n_actor_weights.value) 
            else: 
                #Cholesky covariance matrix
                self.higher_logstd = higher_logstd = tf.get_variable(
                    name='higher_logstd',
                    shape=[n_actor_weights*(n_actor_weights + 1)//2],
                    initializer=tf.initializers.constant(0.))
                pdparam = tf.concat([higher_mean, 
                                    higher_logstd], axis=0)
                self.pdtype = pdtype = CholeskyGaussianPdType(
                    n_actor_weights.value) 

        #Sample actor weights
        self.pd = pdtype.pdfromflat(pdparam)
        sampled_actor_params = self.pd.sample()
        symm_sampled_actor_params = self.pd.sample_symmetric()
        self._sample_symm_actor_params = U.function(
            [],list(symm_sampled_actor_params))
        self._sample_actor_params = U.function([], [sampled_actor_params])
            
        #Assign actor weights
        with tf.variable_scope('actor') as scope:
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)

            self._use_sampled_actor_params = U.assignFromFlat(actor_params,
                                                         sampled_actor_params)

        #Act
        action = actor_mean
        self._act = U.function([ob],[action])

        #Higher policy weights
        with tf.variable_scope('higher') as scope:
            self._higher_params = higher_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name) 
            self.flat_higher_params = tf.concat([tf.reshape(w, [-1]) for w in \
                                            self._higher_params], axis=0) #flatten
            self._n_higher_params = self.flat_higher_params.shape[0]

        #Batch PGPE
        self._actor_params_in = actor_params_in = \
                U.get_placeholder(name='actor_params_in',
                                  dtype=tf.float32,
                                  shape=[batch_length] + [n_actor_weights])
        self._rets_in = rets_in = U.get_placeholder(name='returns_in',
                                                  dtype=tf.float32,
                                                  shape=[batch_length])
        self._logprobs = logprobs = self.pd.logp(actor_params_in)
        pgpe_times_n = U.flatgrad(logprobs*rets_in, higher_params)
        self._get_pgpe_times_n = U.function([actor_params_in, rets_in],
                                            [pgpe_times_n])

        #One-episode PGPE
        #Used N times to compute the baseline -> can we do better?
        self._one_actor_param_in = one_actor_param_in = U.get_placeholder(
                                    name='one_actor_param_in',
                                    dtype=tf.float32,
                                    shape=[n_actor_weights])
        one_logprob = self.pd.logp(one_actor_param_in)
        score = U.flatgrad(one_logprob, higher_params)
        score_norm = tf.norm(score)
        self._get_score = U.function([one_actor_param_in], [score])
        self._get_score_norm = U.function([one_actor_param_in], [score_norm])

        #Batch off-policy PGPE
        self._probs = tf.exp(logprobs) 
        self._behavioral = None
    
        #One episode off-PGPE 
        self._one_prob = tf.exp(one_logprob)
        
        #Renyi computation
        self._det_sigma = tf.exp(tf.reduce_sum(self.higher_logstd))

        #Fisher computation (diagonal case)
        mean_fisher_diag = tf.exp(-self.higher_logstd)
        cov_fisher_diag = 2*tf.exp(2*self.higher_logstd)
        self._fisher_diag = tf.concat([mean_fisher_diag, mean_fisher_diag * 0. +
                               cov_fisher_diag], axis=0)

        
    #Black box usage
    def act(self, ob, resample=False):
        """
        Sample weights for the actor network, then sample action(s) from the 
        resulting actor depending on state(s)
           
        Params:
               ob: current state, or a list of states
               resample: whether to resample actor params before acting
        """
        if hasattr(self, "ob_rms"):
            self.ob_rms.update(ob) # update running mean/std for policy
        
        if resample:
            self.resample()

        action =  self._act(np.atleast_2d(ob))[0]
        return action

    def resample(self):
        """Resample actor params
        
        Returns:
            the sampled actor params
        """
        tf.get_default_session().run(self._use_sampled_actor_params)
        return self.eval_actor_params()
    
    def eval_params(self):
        """Get current params of the higher order policy"""
        return U.GetFlat(self._higher_params)()

    def set_params(self, new_higher_params):
        """Set higher order policy parameters from flat sequence"""
        U.SetFromFlat(self._higher_params)(new_higher_params)

    def seed(self, seed):
        if seed is not None:
            set_global_seeds(seed)

    #Direct actor policy manipulation
    def draw_actor_params(self):
        """Sample params for an actor (without using them)"""
        sampled_actor_params = self._sample_actor_params()[0]
        return sampled_actor_params

    def draw_symmetric_actor_params(self):
        return tuple(self._sample_symm_actor_params())

    def eval_actor_params(self):
        """Get actor params as last assigned"""
        with tf.variable_scope(self.scope+'/actor') as scope:
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)
        return U.GetFlat(actor_params)()    

    def set_actor_params(self, new_actor_params):
        """Manually set actor policy parameters from flat sequence"""
        with tf.variable_scope(self.scope+'/actor') as scope:
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)
            U.SetFromFlat(actor_params)(new_actor_params)

    #Distribution properties
    def renyi(self, other, order=2, exponentiate=False):
        """Renyi divergence 
            Special case: order=1 is kl divergence
        
        Params:
            other: policy to evaluate the distance from
            order: order of the Renyi divergence
            exponentiate: if true, actually returns e^Renyi(self||other)
        """
        if order==1:
            result = self.pd.kl(other.pd)
        elif order>=2:
            result = self.pd.renyi(other.pd, alpha=order) 
        else:
            raise ValueError('Order must be >= 1')
        
        if exponentiate:
            result = tf.exp(result)
        return U.function([], [result])()[0]

    def eval_fisher(self, return_diagonal=True):
        if not self.diagonal or not return_diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return np.ravel(U.function([],[self._fisher_diag])()[0])

    def fisher_product(self, x):
        if not self.diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return x/self.eval_fisher()

    #Gradient computation
    def eval_gradient(self, actor_params, rets, use_baseline=True,
                      behavioral=None):
        """
        Compute PGPE policy gradient given a batch of episodes

        Params:
            actor_params: list of actor parameters (arrays), one per episode
            rets: flat list of total [discounted] returns, one per episode
            use_baseline: wether to employ a variance-minimizing baseline 
                (may be more efficient without)
            behavioral: higher-order policy used to collect data (off-policy
                case). If None, the present policy is assumed to be the 
                behavioral(on-policy case)

        References:
            Optimal baseline for PGPE: Zhao, Tingting, et al. "Analysis and
            improvement of policy gradient estimation." Advances in Neural
            Information Processing Systems. 2011.

        """ 
        assert rets and len(actor_params)==len(rets)
        batch_size = len(rets)
        
        if not behavioral:
            #On policy
            if not use_baseline:
                #Without baseline (more efficient)
                pgpe_times_n = np.ravel(self._get_pgpe_times_n(actor_params, rets)[0])
                return pgpe_times_n/batch_size
            else:
                #With optimal baseline
                rets = np.array(rets)
                scores = np.zeros((batch_size, self._n_higher_params))
                score_norms = np.zeros(batch_size)
                for (theta, i) in zip(actor_params, range(batch_size)):
                    scores[i] = self._get_score(theta)[0]
                    score_norms[i] = self._get_score_norm(theta)[0]
                b = np.sum(rets * score_norms**2) / np.sum(score_norms**2)
                pgpe = np.mean(((rets - b).T * scores.T).T, axis=0)
                return pgpe
        else:
            #Off-policy
            if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
            if not use_baseline:
                #Without baseline (more efficient)
                off_pgpe_times_n = np.ravel(self._get_off_pgpe_times_n(actor_params,
                                                              rets)[0])
                return off_pgpe_times_n/batch_size
            else:
                #With optimal baseline
                rets = np.array(rets)
                scores = np.zeros((batch_size, self._n_higher_params))
                score_norms = np.zeros(batch_size)
                for (theta, i) in zip(actor_params, range(batch_size)):
                    scores[i] = self._get_score(theta)[0]
                    score_norms[i] = self._get_score_norm(theta)[0]
                iws = np.ravel(self._get_iws(actor_params)[0])
                b = np.sum(rets * iws**2 * score_norms**2)/ np.sum(iws**2 *
                                                                   score_norms**2)
                pgpe = np.mean(((rets - b).T * scores.T).T, axis=0)
                return pgpe
                

    def eval_natural_gradient(self, actor_params, rets, use_baseline=True,
                      behavioral=None):
        """
        Compute PGPE policy gradient given a batch of episodes

        Params:
            actor_params: list of actor parameters (arrays), one per episode
            rets: flat list of total [discounted] returns, one per episode
            use_baseline: wether to employ a variance-minimizing baseline 
                (may be more efficient without)
            behavioral: higher-order policy used to collect data (off-policy
                case). If None, the present policy is assumed to be the 
                behavioral(on-policy case)

        References:
            Optimal baseline for PGPE: Zhao, Tingting, et al. "Analysis and
            improvement of policy gradient estimation." Advances in Neural
            Information Processing Systems. 2011.

        """ 
        assert rets and len(actor_params)==len(rets)
        batch_size = len(rets)
        fisher = self.eval_fisher() + 1e-24
        
        if not behavioral:
            #On policy
            if not use_baseline:
                #Without baseline (more efficient)
                pgpe_times_n = np.ravel(self._get_pgpe_times_n(actor_params, rets)[0])
                grad = pgpe_times_n/batch_size
                if self.diagonal:
                    return grad/fisher
                else: 
                    raise NotImplementedError #TODO: full on w/o baseline
            else:
                #With optimal baseline
                if self.diagonal:
                    rets = np.array(rets)
                    scores = np.zeros((batch_size, self._n_higher_params))
                    score_norms = np.zeros(batch_size)
                    for (theta, i) in zip(actor_params, range(batch_size)):
                        scores[i] = self._get_score(theta)[0]
                        score_norms[i] = np.linalg.norm(scores[i]/fisher)
                    b = np.sum(rets * score_norms**2) / np.sum(score_norms**2)
                    npgpe = np.mean(((rets - b).T * scores.T).T, axis=0)/fisher
                    return npgpe
                else:
                    raise NotImplementedError #TODO: full on with baseline
        else:
            #Off-policy
            if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
            if not use_baseline and self.diagonal:
                #Without baseline (more efficient)
                off_pgpe_times_n = np.ravel(self._get_off_pgpe_times_n(actor_params,
                                                              rets)[0])
                grad = off_pgpe_times_n/batch_size
                return grad/fisher
            else:
                raise NotImplementedError #TODO: full off with baseline, diagonal off with baseline
    
    def eval_bound(self, actor_params, rets, behavioral, delta):
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
        batch_size = len(rets)
        penal_coeff = sts.t.ppf(1 - delta, batch_size - 1) / np.sqrt(batch_size)

        return self._get_bound_and_grad(actor_params, rets, penal_coeff)
             
    
    def _build_iw_graph(self, behavioral):
        #Batch
        unn_iws = self._probs/behavioral._probs
        iws = unn_iws/tf.reduce_sum(unn_iws)
        self._get_unn_iws = U.function([self._actor_params_in], [unn_iws])
        self._get_iws = U.function([self._actor_params_in], [iws])
        J_hat, J_var = tf.nn.moments(self._rets_in * iws)
        self._penal_coeff = tf.placeholder(name='penal_coeff', dtype=tf.float32, shape=[])
        bound = self.J_hat - tf.sqrt(self.J_var) * self.penal_coeff
        bound_grad = U.flatgrad(bound)
        off_pgpe_times_n = U.flatgrad((tf.stop_gradient(iws) * 
                                             self._logprobs * 
                                             self._rets_in), 
                                            self._higher_params)
                    
        self._get_off_pgpe_times_n = U.function([self._actor_params_in,
                                                self._rets_in],
                                                [off_pgpe_times_n])
        self._get_bound_and_grad = U.function([self._actor_params_in, self._rets_in, self._penal_coeff],[bound, bound_grad])
