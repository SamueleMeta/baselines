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

class MlpActor(object):
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


class PerWeightPeMlpPolicy(object):
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
            self.actor = MlpActor(ob_space, ac_space, hid_layers,
                  deterministic,
                  use_bias, use_critic)
            
            self.size = n_actor_params = ob_space.shape[0]*hid_layers[0] + \
                                sum(hid_layers[i]*hid_layers[i+1] for i in range(len(hid_layers) - 1)) + \
                                hid_layers[-1]*ac_space.shape[0]
            print('Size: %d' % n_actor_params)
            
            self.higher_means = np.random.normal(size=n_actor_params)
            self.higher_logstds = np.zeros(n_actor_params)
            
            self._higher_param = tf.placeholder(name='higher_param', dtype=tf.float32, shape=[2])
            self._actor_params_in = tf.placeholder(name='actor_params_in', dtype=tf.float32, shape=[None, 1])
            pdtype = DiagGaussianPdType(1)
            self.pd = pdtype.pdfromflat(self._higher_param)
            logprobs = self.pd.logp(self._actor_params_in)
            self.probs = tf.exp(logprobs)
            
            self.behavioral = None
            U.initialize()
            self.resample()
        
    def _higher_params(self, i):
        return np.array([self.higher_means[i], self.higher_logstds[i]])
    
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
        actor_params = np.random.multivariate_normal(self.higher_means, np.diag(np.exp(2*self.higher_logstds)))
        self.actor.set_actor_params(actor_params)
        return actor_params
    
    def eval_params(self):
        """Get current params of the higher order policy"""
        return np.concatenate([self.higher_means, self.higher_logstds])

    def set_params(self, new_higher_params):
        """Set higher order policy parameters from flat sequence"""
        means, logstds = np.split(new_higher_params, 2)
        self.higher_means = means
        self.higher_logstds = logstds

    def seed(self, seed):
        if seed is not None:
            set_global_seeds(seed)

    #TODO:
    
    def eval_fisher(self):
        mean_fisher = np.exp(-2*self.higher_logstds)
        logstd_fisher = 2*np.ones(self.size)
        return np.concatenate((mean_fisher, logstd_fisher))

     
    def eval_bound(self, actor_params, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        batch_size = len(rets)
        ppf = sts.norm.ppf(1 - delta)
        if behavioral is not self.behavioral:
            self.behavioral = behavioral
            self._batch_size = tf.placeholder(name='batch_size', dtype=tf.float32, shape=[])
            self._rmax = tf.placeholder(name='rmax', dtype=tf.float32, shape=[])
            self._ppf = tf.placeholder(name='ppf', dtype=tf.float32, shape=[])
            self._rets_in = tf.placeholder(name='rets_in', dtype=tf.float32, shape=[None])
            renyi = self.pd.renyi(behavioral.pd)
            unn_iws = self.probs/behavioral.probs
            iws = unn_iws/tf.reduce_sum(unn_iws)
            _bound = tf.reduce_sum(self._rets_in*iws) - self._ppf*tf.exp(0.5*renyi)*self._rmax/self._batch_size
            self._get_bound = U.function([self._higher_param, behavioral._higher_param, self._actor_params_in,
                                          behavioral._actor_params_in, self._rets_in,
                                          self._batch_size, self._rmax, self._ppf], [_bound])
            self._get_bound_grad = U.function([self._higher_param, behavioral._higher_param,
                                               self._actor_params_in, behavioral._actor_params_in, self._rets_in,
                                          self._batch_size, self._rmax, self._ppf], [tf.gradients(_bound, self._higher_param)])
        
        bound = []
        for i in range(self.size):
            higher_param = np.array([self.higher_means[i], self.higher_logstds[i]])
            other_higher_param = np.array([behavioral.higher_means[i], behavioral.higher_logstds[i]])
            actor_params_in = np.array(actor_params)[:, i]
            actor_params_in = np.expand_dims(actor_params_in, -1)
            bound.append(self._get_bound(higher_param, other_higher_param,
                                         actor_params_in, actor_params_in, rets, batch_size, rmax, ppf)[0])
        return np.max(bound)
            
    
    
            
    
    def eval_bound_and_grad(self, actor_params, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        batch_size = len(rets)
        ppf = sts.norm.ppf(1 - delta)
        bound = self.eval_bound(actor_params, rets, behavioral, delta, normalize,
                   rmax)
        mean_grads, logstd_grads = [], []
        for i in range(self.size):
            higher_param = np.array([self.higher_means[i], self.higher_logstds[i]])
            other_higher_param = np.array([behavioral.higher_means[i], behavioral.higher_logstds[i]])
            actor_params_in = np.array(actor_params)[:, i]
            actor_params_in = np.expand_dims(actor_params_in, -1)
            grad = self._get_bound_grad(higher_param, other_higher_param,
                                         actor_params_in, actor_params_in, rets, batch_size, rmax, ppf)[0][0]
            mean_grads.append(grad[0])
            logstd_grads.append(grad[1])
        return bound, np.concatenate((mean_grads, logstd_grads))