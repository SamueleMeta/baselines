from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np
import time
import scipy.stats as sts


class MlpPolicy(object):
    """Gaussian policy with critic, based on multi-layer perceptron"""
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        #with tf.device('/cpu:0'):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            U.initialize()

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, use_bias=True):
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
        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        #Actor
        with tf.variable_scope('pol'):
            last_out = obz
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
        self._act = U.function([stochastic, ob], [ac, self.vpred])

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
        flat_weights = tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)
        self.n_weights = flat_weights.shape[0].value
        self.score = score = U.flatgrad(self.logprobs, weights) # \nabla\log p(\tau)
        self.fisher = tf.einsum('i,j->ij', score, score)        
      
        #Fisher2
        self.built_hesslog = False
        """
        hessian_rows = [U.flatgrad(score, weights, grad_ys=hessian_mask[i, :]) for i in range(n_weights)]
        self.hesslog = tf.stack(hessian_rows, axis=0)
        #"""
      
        
    def build_hesslog(self):
        checkpoint = time.time()
        hessian_mask = tf.eye(self.n_weights)
        self.hesslog = U.flatgrad(self.score, self.weights, grad_ys=hessian_mask[0, :])
        self.hesslog = tf.expand_dims(self.hesslog, axis=1)
        #"""
        for i in range(1,self.n_weights):
            print('Hessian row', i, '/', self.n_weights)
            new_row = U.flatgrad(self.score, self.weights, grad_ys=hessian_mask[i])
            new_row = tf.expand_dims(new_row, axis=1)
            self.hesslog = tf.concat([self.hesslog, new_row], axis=1)
        #"""
        print('fisher2 compile time:', time.time() - checkpoint)
        self.built_hesslog = True
        
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
    def eval_performance(self, states, actions, rewards, lens_or_batch_size=1, horizon=None, behavioral=None, per_decision=False, gamma=.99):        
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
            average episodic performance J_hat, 
            variance of episodic performance Var_J, 
            gradient of J_hat wrt actor parameters,
            gradient of Var_J wrt actor parameters
        """
        #Prepare data
        checkpoint = time.time()
        assert len(rewards) > 0
        assert len(states)==len(actions)==len(rewards)
        if type(lens_or_batch_size) is list:
            lens = lens_or_batch_size
            assert sum(lens) > 0
            batch_size = len(lens)
            if horizon is None:
                horizon = max(lens)
            assert all(np.array(lens) <= horizon)
        else:
            assert type(lens_or_batch_size) is int
            batch_size = lens_or_batch_size
            assert len(rewards)%batch_size == 0
            if horizon is None:
                horizon = len(rewards)/batch_size
            lens = [horizon] * batch_size
        _states, _actions, _rewards, _mask = self._prepare_data(states, actions, rewards, lens, horizon)
        #print('Prepare time:', time.time() - checkpoint)
        
        #Build performance evaluation graph
        #checkpoint = time.time()        
        fun_0, fun_1 = self._build_performance(batch_size, horizon, behavioral, per_decision)
        #print('Compile time:', time.time() - checkpoint)
        
        #Evaluate performance
        #checkpoint = time.time()
        result_0 = fun_0(_states, _actions, _rewards, gamma, _mask)
        #print('Run time:', time.time() - checkpoint)
        print('Performance eval time:', time.time() - checkpoint)
        
        result_0 = list(map(np.asscalar, result_0))
        def result_1():
            return fun_1(_states, _actions, _rewards, gamma, _mask)
        result = result_0 + [result_1] #list concat
        return tuple(result)
    
    def _prepare_data(self, states, actions, rewards, lens, horizon, do_pad=True, do_concat=True):
        no_of_samples = sum(lens)       
        mask = np.ones(no_of_samples) if do_pad else None
        
        indexes = np.cumsum(lens)
        to_resize = [states, actions, rewards, mask]
        to_resize = [x for x in to_resize if x is not None]
        resized = []
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

    def _build_performance(self, batch_size, horizon, behavioral, per_decision):
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
        
        return (U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [avg_J, var_J]), 
                    U.function([self.ob, self.ac_in, self.rew, self.gamma, self.mask], [grad_avg_J, grad_var_J])) 
    
    def eval_fisher(self, states, actions, lens_or_batch_size, horizon=None, behavioral=None):
        """
        Fisher information matrix of actor parameters, possibly off-policy
        
        Params:
            states, actions as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a) pairs
        """
        checkpoint = time.time()
        assert len(states) > 0
        assert len(states)==len(actions)
        if type(lens_or_batch_size) is list:
            lens = lens_or_batch_size
            assert sum(lens) > 0
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
            lens = [horizon] * batch_size
        _states, _actions = self._prepare_data(states, 
                                                      actions, 
                                                      None, 
                                                      lens, 
                                                      horizon,
                                                      do_pad=False,
                                                      do_concat=False)
        print('Fisher prep time:', time.time() - checkpoint)
        checkpoint = time.time()
        fisher = self.fisher
        #with tf.device('/cpu:0'):
        if behavioral is not None:
            log_ratios = self.logprobs - behavioral.pd.logp(self.ac_in)
            iw = tf.exp(tf.reduce_sum(log_ratios))
            fisher = tf.multiply(iw, fisher)
        
        fun =  U.function([self.ob, self.ac_in], [fisher])
        fisher_samples = np.array([fun(s, a)[0] for (s,a) in zip(_states, _actions)]) #one call per EPISODE
        print('Fisher eval time:', time.time() - checkpoint)
        return np.mean(fisher_samples, axis=0)

    def eval_fisher2(self, states, actions, lens_or_batch_size, horizon=None, behavioral=None):
        """
        Fisher information matrix of actor parameters, possibly off-policy
        
        Params:
            states, actions as lists, flat wrt time
            lens_or_batch_size: list with episode lengths or scalar representing the number of (equally long) episodes
            horizon: max task horizon
            behavioral: policy used to collect (s, a) pairs
        """
        checkpoint = time.time()
        assert len(states) > 0
        assert len(states)==len(actions)
        if type(lens_or_batch_size) is list:
            lens = lens_or_batch_size
            assert sum(lens) > 0
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
            lens = [horizon] * batch_size
        _states, _actions = self._prepare_data(states, 
                                                      actions, 
                                                      None, 
                                                      lens, 
                                                      horizon,
                                                      do_pad=False,
                                                      do_concat=True)
        print('Fisher prep time:', time.time() - checkpoint)
        checkpoint = time.time()
        if not self.built_hesslog:
            self.build_hesslog()
        fisher = - self.hesslog / batch_size
        
        fun =  U.function([self.ob, self.ac_in], [fisher])
        print('Fisher eval time:', time.time() - checkpoint)
        return fun(_states, _actions)[0]
    
    def student_t_bound(self, states, actions, rewards, lens_or_batch_size=1, delta=.2, horizon=None, behavioral=None, per_decision=False, gamma=.99):
        if type(lens_or_batch_size) is list:
            batch_size = len(lens_or_batch_size)
        else:
            assert type(lens_or_batch_size) is int
            batch_size = lens_or_batch_size
            
        J_hat, var_J, _ = self.eval_performance(states, actions, rewards, lens_or_batch_size, horizon, behavioral, per_decision, gamma)
        bound = J_hat - sts.t.ppf(1 - delta, batch_size - 1) / np.sqrt(batch_size) * np.sqrt(var_J)
        return bound


    #Weight manipulation
    def eval_param(self):
        """"Policy parameters (numeric,flat)"""
        with tf.variable_scope(self.scope+'/pol') as vs:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)
        return U.GetFlat(var_list)()

    def get_param(self):
        """Policy parameters (symbolic, nested)"""
        with tf.variable_scope(self.scope+'/pol') as vs:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=vs.name)

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