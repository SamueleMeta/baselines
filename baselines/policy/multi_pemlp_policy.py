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
        actor_params = np.random.multivariate_normal(self.higher_means, np.diag(self.higher_logstds))
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
    
    def eval_renyi(self, other, order=2):
        """Renyi divergence"""
        assert type(other) is PerWeightPeMlpPolicy
        return 1.
        
    def eval_fisher(self):
        return 2*np.ones(self.size*2)

     
    def eval_bound(self, actor_params, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        actor_params = np.array(actor_params)
        return np.zeros(self.size)
    
    def eval_bound_and_grad(self, actor_params, rets, behavioral, delta=0.2, normalize=True,
                   rmax=None):
        actor_params = np.array(actor_params)
        return np.zeros(self.size), np.zeros(2*self.size)