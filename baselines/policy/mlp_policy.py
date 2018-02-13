from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

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
        self.logprobs = self.pd.logp(self.ac_in)

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def eval_performance(self,states,actions,rewards,lens=1,behavioral=None,per_decision=False,gamma=.99):
        """Performance under the policy

            Params:
            states: flat list of states
            actions: flat list of actions
            rewards: flat list of rewards
            lens: list->episode lengths, number->batch_size
            behavioral: policy w/ which states, actions, rewards were sampled
            per_decision: whether to use pd importance weights instead of
                            regular ones
            gamma: discount factor
        """

        #Prepare s,a,r
        import numpy as np
        states = np.array(states)
        if states.ndim==1:
            states = np.expand_dims(states,axis=1)
        actions = np.array(actions)
        if actions.ndim==1:
            actions = np.expand_dims(actions,axis=1)
        rewards = np.array(rewards)
        if rewards.ndim==1:
            rewards = np.expand_dims(rewards,axis=1)
        assert len(states)==len(actions)==len(rewards)

        batch_size = len(lens)
        rews_by_episode = tf.split(self.rew,lens)
        rews_by_episode = tf.stack(self._fill(rews_by_episode))
        horizon = rews_by_episode.shape[1]

        disc = self.gamma*tf.ones([batch_size,horizon,1])
        disc = tf.cumprod(disc,axis=1,exclusive=True)
        disc_rews = rews_by_episode*disc

        if behavioral is None:
            weighted_rets = tf.reduce_sum(disc_rews,axis=1)
        else:
            log_ratios = self.logprobs - behavioral.pd.logp(self.ac_in)
            log_ratios_by_episode = tf.split(log_ratios,lens)
            log_ratios_by_episode = tf.stack(self._fill(log_ratios_by_episode))

            if per_decision:
                iw = tf.exp(tf.cumsum(log_ratios_by_episode, axis=1))
                iw = tf.expand_dims(iw,axis=2)
                weighted_rets = tf.reduce_sum(tf.multiply(disc_rews,iw),axis=1)
            else:
                iw = tf.exp(tf.reduce_sum(log_ratios_by_episode,axis=1))
                rets = tf.reduce_sum(disc_rews,axis=1)
                weighted_rets = tf.multiply(rets,iw)
        J_hat = tf.reduce_mean(weighted_rets)
        fun = U.function([self.ob,self.ac_in,self.rew,self.gamma],[J_hat])
        return fun(states,actions,rewards,gamma)[0]

    def _fill(self,tensors,filler=0):
        max_len = max([t.shape[0]] for t in tensors)
        result = []
        for t in tensors:
            padding = filler*tf.ones(shape=[max_len[0] - t.shape[0]] + \
                                list(t.shape)[1:],dtype=tf.float32)
            t = tf.concat([t,padding],axis=0)
            result.append(t)
        return result

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

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
