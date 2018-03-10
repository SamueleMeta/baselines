from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np
import scipy.stats as sts
#import time

class MlpPolicy(object):
    """Gaussian policy with critic, based on multi-layer perceptron"""
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        #with tf.device('/cpu:0'):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            U.initialize()

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, use_bias=True, use_critic=True):
        """Params:
            ob_space: task observation space
            ac_space : task action space
            hid_size: width of hidden layers
            num_hid_layers: depth
            gaussian_fixed_var: True->separate parameter for logstd, False->two-headed mlp
            use_bias: whether to include bias in neurons
        """
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        #Critic
        if use_critic:
            with tf.variable_scope('vf'):
                obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        #Actor
        with tf.variable_scope('pol'):
            last_out = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                                                      name='fc%i'%(i+1),
                                                      kernel_initializer=U.normc_initializer(1.0),use_bias=use_bias))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2,
                                       name='final',
                                       kernel_initializer=U.normc_initializer(0.01),
                                       use_bias=use_bias)
                logstd = tf.get_variable(name="pol_logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0],
                                          name='final',
                                          kernel_initializer=U.normc_initializer(0.01))

        #Acting
        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        if use_critic:
            self._act = U.function([stochastic, ob], [ac, self.vpred])
        else:
            self._act = U.function([stochastic, ob], [ac, tf.zeros(1)])

        #Evaluating
        self.ob = ob
        self.ac_in = U.get_placeholder(name="ac_in", dtype=tf.float32,
                                       shape=[sequence_length] +
                                       list(ac_space.shape))
        self.gamma = U.get_placeholder(name="gamma",
                                        dtype=tf.float32,shape=[])
        self.rew = U.get_placeholder(name="rew", dtype=tf.float32,
                                       shape=[sequence_length]+[1])
        self.logprobs = self.pd.logp(self.ac_in) #  [\log\pi(a|s)]
        
        #Fisher
        with tf.variable_scope('pol') as vs:
            self.weights = weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
        self.flat_weights = flat_weights = tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
        self.n_weights = flat_weights.shape[0].value
        self.score = score = U.flatgrad(self.logprobs, weights) # \nabla\log p(\tau)
        self.fisher = tf.einsum('i,j->ij', score, score)        
      
        #Performance graph initializations
        self._batch_size = -1
    
    #Acting    
    def act(self, stochastic, ob):
        """
        Actions sampled from the policy
           
        Params:
               stochastic: use noise
               ob: current state
        """
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    
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
        Gradient of student-t bound
        
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


    #Used by original implementation
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []