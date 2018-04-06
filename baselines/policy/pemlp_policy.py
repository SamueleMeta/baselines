from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import DiagGaussianPdType
import numpy as np
from baselines.common import set_global_seeds

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

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers,
              deterministic=True, diagonal=True,
              use_bias=True, standardize_input=True, use_critic=False, 
              seed=None):
        """Params:
            ob_space: task observation space
            ac_space : task action space
            hid_size: width of hidden layers of action policy networks
            num_hid_layers: depth of action policy networks
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
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        #Actor (N.B.: weight initialization is irrelevant)
        with tf.variable_scope('actor'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz if standardize_input else ob
            for i in range(num_hid_layers):
                #Mlp feature extraction
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                                                      name='fc%i'%(i+1),
                                                      kernel_initializer=U.normc_initializer(1.0),use_bias=use_bias))
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
                                               initializer=tf.initializers.random_normal())
            if diagonal:
                #Diagonal covariance matrix; all stds initialized to 0
                self.higher_logstd = higher_logstd = tf.get_variable(name='higher_logstd',
                                               shape=[n_actor_weights],
                                               initializer=tf.initializers.constant(0.))
            else: 
                raise NotImplementedError #Currently supports only diagonal higher order policies
        
        #Sample actor weights
        pdparam = tf.concat([higher_mean, higher_mean * 0. + \
                               higher_logstd], axis=0)
        self.pdtype = pdtype = DiagGaussianPdType(2 * n_actor_weights.value) 
        self.pd = pdtype.pdfromflat(pdparam)
        sampled_actor_params = self.pd.sample()
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
        self._probs = probs = tf.exp(logprobs) 
        self._behavioral = None
    
        #One episode off-PGPE 
        self._one_prob = one_prob = tf.exp(one_logprob)
        
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
    def renyi(self, other, order=2, exponentiate=True):
        """Renyi divergence
        
        Params:
            other: policy to evaluate the distance from
            order: order of the Renyi divergence
            exponentiate: if true, actually returns e^Renyi(self, other)
        """
        if order<2:
            raise NotImplementedError('Only order>=2 currently available')
        if not self.diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        to_check = (order/tf.exp(self.higher_logstd) +
                    (1 - order)/tf.exp(other.higher_logstd))
        if not (U.function([],
                           [to_check])()[0] > 0).all():
            raise ValueError('Conditions on standard deviations not met')
        det0 = self._det_sigma
        det1 = other._det_sigma
        mix = (order*tf.exp(self.higher_logstd) + 
                   (1 - order)*tf.exp(other.higher_logstd))
        det_mix = tf.reduce_prod(mix)
        renyi = (order/2 * tf.reduce_sum((self.higher_mean -
                                           other.higher_mean)**2/mix) -
                            1./(2*order - 1)*(tf.log(det_mix) - 
                                              (1-order)*tf.log(det0) -
                                              order*tf.log(det1)))
        result = tf.exp(renyi) if exponentiate else renyi
        return U.function([], [result])()[0]

    def eval_fisher(self, return_diagonal=True):
        if not self.diagonal or not return_diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return np.ravel(U.function([],[self._fisher_diag])()[0])

    def fisher_product(self, theta):
        if not self.diagonal or not return_diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return theta/self.eval_fisher()

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
                


    def _build_iw_graph(self, behavioral):
        #Batch
        iws = self._probs/behavioral._probs
        self._get_iws = U.function([self._actor_params_in], [iws])
        off_pgpe_times_n = U.flatgrad((tf.stop_gradient(iws) * 
                                             self._logprobs * 
                                             self._rets_in), 
                                            self._higher_params)
        self._get_off_pgpe_times_n = U.function([self._actor_params_in,
                                                self._rets_in],
                                                [off_pgpe_times_n])
        
'''
    #Divergence
    def eval_renyi(self, states, other, order=2):
        """Exponentiated Renyi divergence exp(Renyi(self, other)) for each state
    
    Params:
            states: flat list of states
            other: other policy
            order: order \alpha of the divergence
        """
        if order<2:
            raise NotImplementedError('Only order>=2 is currently supported')
        to_check = order/tf.exp(self.logstd) + (1 - order)/tf.exp(other.logstd)
        if not all(U.function([self.ob],[to_check])(states)[0] > 0):
            raise ValueError('Conditions on standard deviations are not met')
        detSigma = tf.exp(tf.reduce_sum(self.logstd))
        detOtherSigma = tf.exp(tf.reduce_sum(other.logstd))
        mixSigma = order*tf.exp(self.logstd) + (1 - order) * tf.exp(other.logstd)
        detMixSigma = tf.reduce_prod(mixSigma)
        renyi = order/2 * (self.mean - other.mean)/mixSigma*(self.mean - other.mean) - \
            1./(2*(order - 1))*(tf.log(detMixSigma) - (1-order)*tf.log(detSigma) - order*tf.log(detOtherSigma))
        e_renyi = tf.exp(renyi)
        fun = U.function([self.ob],[e_renyi])
        return fun(states)[0]
        
    
    #Performance evaluation
    def eval_J(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99):        
        """
        Performance evaluation, possibly off-policy
        
        Params:
            states, actions, rewards as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
            per_decision: whether to use Per-Decision IS in place of regular episodic IS
            gamma: discount factor
        
        Returns:
            sample variance of episodic performance Var_J_hat, 
        """
        #Prepare data
        batch_size, horizon, _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens_or_batch_size, horizon)
        
        #Build performance evaluation graph (lazy)
        assert horizon>0 and batch_size>0
        self._build(batch_size, horizon, behavioral, per_decision)
        
        #Evaluate performance stats
        result = self._get_avg_J(_states, _actions, _rewards, gamma, _mask)[0]
        return np.asscalar(result)

    def eval_var_J(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99):        
        """
        Performance variance evaluation, possibly off-policy
        
        Params:
            states, actions, rewards as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
            per_decision: whether to use Per-Decision IS in place of regular episodic IS
            gamma: discount factor
        
        Returns:
            sample variance of episodic performance J_hat
        """
        #Prepare data
        batch_size, horizon, _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens_or_batch_size, horizon)
        
        #Build performance evaluation graph (lazy)
        assert horizon>0 and batch_size>0
        self._build(batch_size, horizon, behavioral, per_decision)
        
        #Evaluate performance stats
        result = self._get_var_J(_states, _actions, _rewards, gamma, _mask)[0]
        return np.asscalar(result)

    def eval_grad_J(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99):
        """
        Gradients of performance 
        
        Params:
            states, actions, rewards as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
            per_decision: whether to use Per-Decision IS in place of regular episodic IS
            gamma: discount factor
        
        Returns:
            gradient of average episodic performance wrt actor weights, 
        """
        #Prepare data
        batch_size, horizon, _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens_or_batch_size, horizon)
        
        #Build performance evaluation graph (lazy)
        assert batch_size>0
        self._build(batch_size, horizon, behavioral, per_decision)
        
        #Evaluate gradients
        result = self._get_grad_J(_states, _actions, _rewards, gamma, _mask)[0]
        return np.ravel(result)

    def eval_grad_var_J(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99):
        """
        Gradients of performance stats
        
        Params:
            states, actions, rewards as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
            per_decision: whether to use Per-Decision IS in place of regular episodic IS
            gamma: discount factor
        
        Returns:
            gradient of sample variance of episodic performance wrt actor weights 
        """
        #Prepare data
        batch_size, horizon, _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens_or_batch_size, horizon)
        
        #Build performance evaluation graph (lazy)
        assert batch_size>0
        self._build(batch_size, horizon, behavioral, per_decision)
        
        #Evaluate gradients
        result = self._get_grad_var_J(_states, _actions, _rewards, gamma, _mask)[0]
        return np.ravel(result)

    def eval_bound(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99, delta=.2):
        """
        Student-t bound on performance
        
        Params:
            states, actions, rewards as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
            per_decision: whether to use Per-Decision IS in place of regular episodic IS
            gamma: discount factor
            delta: 1 - confidence
        """
        #Prepare data
        batch_size, horizon, _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens_or_batch_size, horizon)
        
        #Build performance evaluation graph (lazy)
        assert horizon>0 and batch_size>0
        self._build(batch_size, horizon, behavioral, per_decision, delta)
        
        #Evaluate bound
        return np.asscalar(self._get_bound(_states, _actions, _rewards, gamma, _mask)[0])
        
    def eval_bound_grad(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99, delta=.2):
        """
        
        
        Params:
            states, actions, rewards as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
            per_decision: whether to use Per-Decision IS in place of regular episodic IS
            gamma: discount factor
            delta: 1 - confidence
        """
        #Prepare data
        batch_size, horizon, _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens_or_batch_size, horizon)
        
        #Build performance evaluation graph (lazy)
        assert horizon>0 and batch_size>0
        self._build(batch_size, horizon, behavioral, per_decision, delta)
        
        #Evaluate gradient
        return np.ravel(self._get_bound_grad(_states, _actions, _rewards, gamma, _mask)[0])
    
    def _prepare_data(self, states, actions, rewards, lens_or_batch_size, horizon, do_pad=True, do_concat=True):
        assert len(states) > 0
        assert len(states)==len(actions)
        if actions is not None:
            assert len(actions)==len(states)
        if type(lens_or_batch_size) is list:
            lens = lens_or_batch_size    
            no_of_samples = sum(lens)
            assert no_of_samples > 0
            batch_size = len(lens)
            if horizon is None:
                horizon = max(lens)
            assert all(np.array(lens) <= horizon)
        else:
            assert type(lens_or_batch_size) is int
            batch_size = lens_or_batch_size
            assert len(states)%batch_size == 0
            if horizon is None:
                horizon = len(states)/batch_size
            no_of_samples = horizon * batch_size
            lens = [horizon] * batch_size
               
        mask = np.ones(no_of_samples) if do_pad else None
        
        indexes = np.cumsum(lens)
        to_resize = [states, actions, rewards, mask]
        to_resize = [x for x in to_resize if x is not None]
        resized = [batch_size, horizon]
        for v in to_resize:
            v = np.array(v[:no_of_samples])
            if v.ndim == 1:
                v = np.expand_dims(v, axis=1)
            v = np.split(v, indexes, axis=0)
            if do_pad:
                padding_shapes = [tuple([horizon - m.shape[0]] + list(m.shape[1:])) for m in v if m.shape[0]>0]
                paddings = [np.zeros(shape, dtype=np.float32) for shape in padding_shapes]
                v = [np.concatenate((m, pad)) for (m, pad) in zip(v, paddings)]
            if do_concat:
                v = np.concatenate(v, axis=0)
            resized.append(v)
        return tuple(resized)

    def _build(self, batch_size, horizon, behavioral, per_decision, delta=.2):
        if batch_size!=self._batch_size or horizon!=self._horizon:
            #checkpoint = time.time()
            self._batch_size = batch_size
            self._horizon = horizon
            self.mask = tf.placeholder(name="mask", dtype=tf.float32, shape=[batch_size*horizon, 1])
            rews_by_episode = tf.split(self.rew, batch_size)
            rews_by_episode = tf.stack(rews_by_episode)
            disc = self.gamma + 0*rews_by_episode
            disc = tf.cumprod(disc, axis=1, exclusive=True)
            disc_rews = rews_by_episode * disc
            
            if behavioral is None:
                avg_J, var_J = tf.nn.moments(tf.reduce_sum(disc_rews, axis=1), axes=[0])
                grad_avg_J = tf.constant(0)
                grad_var_J = tf.constant(0)    
                bound = avg_J - sts.t.ppf(1 - delta, batch_size - 1) / np.sqrt(batch_size) * tf.sqrt(var_J)
                grad_bound = tf.constant(0)
            else:
                log_ratios = self.logprobs - behavioral.pd.logp(self.ac_in)
                log_ratios = tf.expand_dims(log_ratios, axis=1)
                log_ratios = tf.multiply(log_ratios, self.mask)
                log_ratios_by_episode = tf.split(log_ratios, batch_size)
                log_ratios_by_episode = tf.stack(log_ratios_by_episode)
                if per_decision:
                    iw = tf.exp(tf.cumsum(log_ratios_by_episode, axis=1))
                    #iw = tf.expand_dims(iw,axis=2)
                    weighted_rets = tf.reduce_sum(tf.multiply(disc_rews,iw), axis=1)
                else:
                    iw = tf.exp(tf.reduce_sum(log_ratios_by_episode, axis=1))
                    rets = tf.reduce_sum(disc_rews, axis=1)
                    weighted_rets = tf.multiply(rets, iw)
                
                avg_J, var_J = tf.nn.moments(weighted_rets, axes=[0])
                grad_avg_J = U.flatgrad(avg_J, self.get_param())
                grad_var_J = U.flatgrad(var_J, self.get_param())
                bound = avg_J - sts.t.ppf(1 - delta, batch_size - 1) / np.sqrt(batch_size) * tf.sqrt(var_J)
                grad_bound = U.flatgrad(bound, self.get_param())
            
            self._get_avg_J = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [avg_J])
            self._get_var_J = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [var_J])
            #self._get_performance = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [avg_J, var_J])
            self._get_grad_J = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [grad_avg_J])
            self._get_grad_var_J = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [grad_var_J])
            #self._get_performance_grads = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [grad_avg_J, grad_var_J])
            self._get_bound = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [bound])
            self._get_bound_grad = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [grad_bound])
            self._get_all = U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [avg_J, var_J, grad_avg_J, grad_var_J])
            #print('Recompile time:', time.time() - checkpoint)


    #Fisher    
    def eval_fisher(self, states, actions, lens_or_batch_size, horizon=None, behavioral=None):
        """
        Fisher information matrix
        
        Params:
            states, actions as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a, r) tuples
        """
        #Prepare data
        batch_size, horizon, _states, _actions = self._prepare_data(states, 
                                                      actions, 
                                                      None, 
                                                      lens_or_batch_size, 
                                                      horizon,
                                                      do_pad=False,
                                                      do_concat=False)
        fisher = self.fisher
        with tf.device('/cpu:0'):
            if behavioral is not None:
                log_ratios = self.logprobs - behavioral.pd.logp(self.ac_in)
                iw = tf.exp(tf.reduce_sum(log_ratios))
                fisher = tf.multiply(iw, fisher)
            
        fun =  U.function([self.ob, self.ac_in], [fisher])
        fisher_samples = np.array([fun(s, a)[0] for (s,a) in zip(_states, _actions)]) #one call per EPISODE
        return np.mean(fisher_samples, axis=0)


    #Weight manipulation
    def eval_param(self):
        """"Policy parameters (numeric,flat)"""
        with tf.variable_scope(self.scope+'/pol') as vs:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
        return U.GetFlat(var_list)()    
    
    def get_param(self):
        """Policy parameters (symbolic, nested)"""
        """
        with tf.variable_scope(self.scope+'/pol') as vs:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
        """
        return self.weights

    def set_param(self,param):
        """Set policy parameters to (flat) param"""

        with tf.variable_scope(self.scope+'/pol') as vs:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
            U.SetFromFlat(var_list)(param)
'''
