import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import DiagGaussianPdType
import numpy as np
from baselines.common import set_global_seeds
import scipy.stats as sts

"""References
PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
control." International Conference on Artificial Neural Networks. Springer,
Berlin, Heidelberg, 2008.
"""

class MlpPolicy(object):
    def __init__(self, ob_space, ac_space, hid_layers=[],
              deterministic=True,
              use_bias=True, use_critic=False):
        self._ob = ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        #Critic (normally not used)
        if use_critic:
            with tf.variable_scope('critic'):
                last_out = ob
                for i, hid_size in enumerate(hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        
        #Actor (N.B.: weight initialization is irrelevant)
        with tf.variable_scope('actor') as scope:
            last_out = ob
            for i, hid_size in enumerate(hid_layers):
                #Mlp feature extraction
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                                                      name='fc%i'%(i+1),
                                                      kernel_initializer=U.normc_initializer(1),use_bias=use_bias))
            if deterministic and isinstance(ac_space, gym.spaces.Box):
                #Determinisitc action selection
                self.actor_mean = actor_mean = tf.layers.dense(last_out, ac_space.shape[0],
                                       name='action',
                                       kernel_initializer=U.normc_initializer(0.01),
                                       use_bias=use_bias)
            else: 
                raise NotImplementedError #Currently supports only deterministic action policies
    
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)
            self._set_actor_params = U.SetFromFlat(actor_params)
            self._get_actor_params = U.GetFlat(actor_params)

        #Act
        self._action = action = actor_mean
        self._act = U.function([ob],[action])

    
    def act(self, ob):
        """
        Sample weights for the actor network, then sample action(s) from the 
        resulting actor depending on state(s)
           
        Params:
               ob: current state, or a list of states
        """
        action =  self._act(np.atleast_2d(ob))[0]
        return action
    
    def eval_actor_params(self):
        """Get actor params as last assigned"""
        self._get_actor_params()

    def set_actor_params(self, new_actor_params):
        """Manually set actor policy parameters from flat sequence"""
        self._set_actor_params(new_actor_params)

class PerWeightPePolicy(object):
    def __init__(self, index):
        with tf.variable_scope('pe_agent_' + str(index)):
            batch_length = None
        
            #Gaussian Hyperpolicy
            with tf.variable_scope('higher_params') as scope:
                self.mean = tf.get_variable(name='mean', shape=[1], dtype=tf.float32, initializer=tf.initializers.random_normal(0,1))    
                self.logstd = tf.get_variable(name='logstd', shape=[1], dtype=tf.float32, initializer=tf.initializers.zeros)
                pdparam = tf.concat([self.mean, self.logstd], axis=0)
                pdtype = DiagGaussianPdType(1) 
                self.pd = pd = pdtype.pdfromflat(pdparam)
                
            #Sample weight
            sampled_weight = pd.sample()
            self._sample_weight = U.function([], [sampled_weight])
            
            #Hyperpolicy params
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name) 
            self.flat_params = tf.concat([tf.reshape(w, [-1]) for w in \
                                        self.params], axis=0)
            self._n_params = self.flat_params.shape[0]
            self._set_params = U.SetFromFlat(self.params)
            self._get_params = U.GetFlat(self.params)
    
            #On policy gradient
            self._weights_in = weights_in = \
                    U.get_placeholder(name='weights_in',
                                      dtype=tf.float32,
                                      shape=[batch_length])
            self._rets_in = rets_in = U.get_placeholder(name='returns_in',
                                                      dtype=tf.float32,
                                                      shape=[batch_length])
            
            self._batch_size = batch_size = tf.placeholder(name='batchsize', dtype=tf.float32, shape=[])
            ret_mean, ret_std = tf.nn.moments(rets_in, axes=[0])
            self._get_ret_mean = U.function([self._rets_in], [ret_mean])
            self._get_ret_std = U.function([self._rets_in], [ret_std])
            self._logprobs = logprobs = pd.logp(weights_in)
            pgpe = U.flatgrad(logprobs*rets_in, self.params)/batch_size
            self._get_pgpe = U.function([weights_in, rets_in],
                                                [pgpe])
            
            #Off policy gradient
            self._probs = tf.exp(logprobs) 
            self._behavioral = None
            self._renyi_other = None
            
            #Renyi computation
            self._det_sigma = tf.exp(tf.reduce_sum(self.logstd))
    
            #Fisher computation
            mean_fisher_diag = tf.exp(-2*self.logstd)
            cov_fisher_diag = mean_fisher_diag*0 + 2
            self._fisher_diag = tf.concat([mean_fisher_diag, cov_fisher_diag], axis=0)
            self._get_fisher_diag = U.function([], [self._fisher_diag])
            
            #Other initializations
            self._renyi_other = None
            self._behavioral = None
            
    def sample(self):
        return np.asscalar(self._sample_weight()[0])
    
    def eval_params(self):
        """Get current params of the higher order policy"""
        return self._get_params()

    def set_params(self, new_params):
        """Set higher order policy parameters from flat sequence"""
        self._set_params(new_params)
        
    def eval_renyi(self, other, order=2):
        """Renyi divergence 
            Special case: order=1 is kl divergence
        
        Params:
            other: policy to evaluate the distance from
            order: order of the Renyi divergence
            exponentiate: if true, actually returns e^Renyi(self||other)
        """
        if other is not self._renyi_other:
            print('EXTENDING!!')
            self._renyi_order = tf.placeholder(name='renyi_order', dtype=tf.float32, shape=[])
            self._renyi_other = other
            if order<2:
                raise ValueError('Order must be >= 2')
            else:   
                renyi = self.pd.renyi(other.pd, alpha=self._renyi_order) 
                self._get_renyi = U.function([self._renyi_order], [renyi])

        return self._get_renyi(order)[0]

    def eval_fisher(self):
        return np.ravel(self._get_fisher_diag()[0])
    
    def eval_performance(self, weights, rets, behavioral=None):
        batch_size = len(rets)
        if behavioral is None:
            #On policy
            return self._get_ret_mean(rets)[0], self._get_ret_std(rets)[0]
        else:
            #Off policy
            if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
            return self._get_off_ret_mean(rets, weights)[0], self._get_off_ret_std(rets, weights, batch_size)[0]

    
    def eval_iws(self, weights, behavioral, normalize=True):
        if behavioral is not self._behavioral:
            self._build_iw_graph(behavioral)
            self._behavioral = behavioral        
        if normalize:
            return self._get_iws(weights)[0]
        else:
            return self._get_unn_iws(weights)[0]
    
    def eval_bound(self, weights, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
        batch_size = len(rets)
        ppf = sts.norm.ppf(1 - delta)
        
        if normalize:
            bound = self._get_z_bound(weights, rets, batch_size, ppf, rmax)[0]
        else:
            bound = self._get_unn_z_bound(weights, rets, batch_size, ppf, rmax)[0]    
        return bound
    
    def eval_bound_and_grad(self, weights, rets, behavioral, delta=0.2, normalize=True,
                            rmax=None):
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
        batch_size = len(rets)
        ppf = sts.norm.ppf(1 - delta)
            
        if normalize:
            bound, grad = self._get_z_bound_grad(weights, rets, batch_size, ppf, rmax)
        else:
            bound, grad = self._get_unn_z_bound_grad(weights, rets, batch_size, ppf, rmax)
        return bound, grad

    def _build_iw_graph(self, behavioral):
        print('EXTENDING!!')
        
        #Self-normalized importance weights
        unn_iws = self._probs/behavioral._probs
        iws = unn_iws/tf.reduce_sum(unn_iws)
        self._get_unn_iws = U.function([self._weights_in], [unn_iws])
        self._get_iws = U.function([self._weights_in], [iws])
        
        #Offline performance
        ret_mean = tf.reduce_sum(self._rets_in * iws)
        unn_ret_mean = tf.reduce_mean(self._rets_in*unn_iws)
        self._get_off_ret_mean = U.function([self._rets_in, self._weights_in], [ret_mean])
        
        #Offline gradient
        off_pgpe = U.flatgrad((tf.stop_gradient(iws) * 
                                             self._logprobs * 
                                             self._rets_in), 
                                            self.params)/self._batch_size
                    
        self._get_off_pgpe = U.function([self._weights_in,
                                                self._rets_in],
                                                [off_pgpe])
        
        #Z Bound
        renyi = self.pd.renyi(behavioral.pd)
        renyi = tf.cond(tf.is_nan(renyi), lambda: tf.constant(np.inf), lambda: renyi)
        renyi = tf.cond(renyi<0., lambda: tf.constant(np.inf), lambda: renyi)
        self._ppf = tf.placeholder(name='penal_coeff', dtype=tf.float32, shape=[])
        self._rmax = tf.placeholder(name='R_max', dtype=tf.float32, shape=[])
        z_std = self._rmax * tf.exp(0.5*renyi) / tf.sqrt(self._batch_size)
        z_bound = ret_mean - self._ppf * z_std
        z_bound_grad = U.flatgrad(z_bound, self.params)
        unn_z_bound = unn_ret_mean - self._ppf * z_std
        unn_z_bound_grad = U.flatgrad(unn_z_bound, self.params)
        self._get_z_bound = U.function([self._weights_in, self._rets_in, self._batch_size, self._ppf, self._rmax], 
                                       [z_bound])
        self._get_z_bound_grad = U.function([self._weights_in, self._rets_in, self._batch_size, self._ppf, self._rmax], 
                                            [z_bound, z_bound_grad])
        self._get_unn_z_bound = U.function([self._weights_in, self._rets_in, self._batch_size, self._ppf, self._rmax], 
                                       [unn_z_bound])
        self._get_unn_z_bound_grad = U.function([self._weights_in, self._rets_in, self._batch_size, self._ppf, self._rmax], 
                                            [unn_z_bound, unn_z_bound_grad])
    
class MultiPeMlpPolicy(object):
    """Multi-layer-perceptron policy with independent Gaussian parameter-based exploration"""
    def __init__(self, name, ob_space, ac_space, hid_layers=[],
              deterministic=True,
              use_bias=True, use_critic=False, seed=None):
        #with tf.device('/cpu:0'):
        with tf.variable_scope(name):    
            if seed is not None:
                set_global_seeds(seed)
            self.scope = tf.get_variable_scope().name
            
            #Mlp actor
            self.actor = MlpPolicy(ob_space, ac_space, hid_layers,
                  deterministic,
                  use_bias, use_critic)
            
            n_actor_params = 4
            
            #PE agents
            self.agents = [PerWeightPePolicy(i) for i in range(n_actor_params)]
            U.initialize()
            self.resample()
        
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
            actor_params = self.resample()
            
        action =  self.actor.act(ob)
        return (action, actor_params) if resample else action
    
    def resample(self):
        """Resample actor params
        
        Returns:
            the sampled actor params
        """        
        actor_params = np.array([agent.sample() for agent in self.agents])
        self.actor.set_actor_params(actor_params)
        return actor_params
    
    def eval_params(self):
        """Get current params of the higher order policy"""
        return np.array([agent.eval_params() for agent in self.agents])

    def set_params(self, new_higher_params):
        """Set higher order policy parameters from flat sequence"""
        for i, agent in enumerate(self.agents):
            agent.set_params(new_higher_params[i])

    def seed(self, seed):
        if seed is not None:
            set_global_seeds(seed)

    def eval_renyi(self, other, order=2):
        """Renyi divergence"""
        assert type(other) is MultiPeMlpPolicy
        return np.array([agent.eval_renyi(other_agent, order) for 
                         agent, other_agent in zip(self.agents, other.agents)])
        
    def eval_fisher(self):
        return np.array([agent.eval_fisher() for agent in self.agents])

     
    def eval_bound(self, actor_params, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        actor_params = np.array(actor_params)
        return np.array([agent.eval_bound(self, actor_params[i, :], rets, behavioral, delta, normalize,
                   rmax) for agent, i in enumerate(self.agents)])
    
    def eval_bound_and_grad(self, actor_params, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        actor_params = np.array(actor_params)
        return np.array([agent.eval_bound_and_grad(actor_params[i, :], rets, behavioral.agents[i], delta, normalize,
                   rmax) for i, agent in enumerate(self.agents)])