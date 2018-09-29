from multiprocessing import Process, Queue, Event
import os
import baselines.common.tf_util as U
import time
import sys
from mpi4py import MPI
from baselines.common import set_global_seeds as set_all_seeds
import numpy as np

def traj_segment_function(pi, env, n_episodes, horizon, stochastic):
    '''
    Collects trajectories
    '''

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    env_d = 0
    _tenv = time.time()
    ob = np.array(env.reset())
    env_d += time.time() - _tenv

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    policy_call_counter = []
    policy_cum_time = []
    env_time = []
    total_time = []

    i = 0
    j = 0
    # Time vars
    tt = 0
    pi_d = 0
    pi_c = 0

    _tt = time.time()

    while True:
        prevac = ac
        _tpi = time.time()
        ac, vpred = pi.act(stochastic, ob)
        pi_d += time.time() - _tpi
        pi_c += 1

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if i == n_episodes:
            return {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "mask" : mask,
                    'policy_cum_time': policy_cum_time, 'policy_call_counter': policy_call_counter,
                    "total_time": total_time, 'env_time': env_time}

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        _tenv = time.time()
        ob, rew, new, _ = env.step(ac)
        ob = np.array(ob)
        env_d += time.time() - _tenv

        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1
        if new or j == horizon:
            # Total time
            tt = time.time() - _tt
            total_time.append(tt)

            new = True
            env.done = True

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            # Policy and env time counters
            policy_cum_time.append(pi_d)
            policy_call_counter.append(pi_c)
            env_time.append(env_d)
            pi_d = 0
            pi_c = 0
            env_d = 0
            # Total time reset
            _tt = time.time()

            cur_ep_ret = 0
            cur_ep_len = 0

            _tenv = time.time()
            ob = np.array(env.reset())
            env_d += time.time() - _tenv

            next_t = (i+1) * horizon

            mask[t+1:next_t] = 0.
            acs[t+1:next_t] = acs[t]
            obs[t+1:next_t] = obs[t]

            t = next_t - 1
            i += 1
            j = 0
        t += 1


class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any tensorflow session!
    '''

    def __init__(self, output, input, event, make_env, make_pi, traj_segment_generator, seed):
        super(Worker, self).__init__()
        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_pi = make_pi
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed

    def run(self):

        sess = U.single_threaded_session()
        sess.__enter__()

        env = self.make_env()
        workerseed = self.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_all_seeds(workerseed)
        env.seed(workerseed)
        #print('Worker %s - Creating policy...' % (os.getpid()))
        pi = self.make_pi('pi%s' % os.getpid(), env.observation_space, env.action_space)
        print('Worker %s - Running with seed %s' % (os.getpid(), workerseed))

        while True:
            self.event.wait()
            self.event.clear()
            command, weights = self.input.get()
            if command == 'collect':
                #print('Worker %s - Collecting...' % os.getpid())
                pi.set_parameter(weights)
                samples = self.traj_segment_generator(pi, env)
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                print('Worker %s - Exiting...' % os.getpid())
                env.close()
                sess.close()
                break

class ParallelSampler(object):

    def __init__(self, make_pi, make_env, n_episodes, horizon, stochastic, n_workers=-1, seed=0):
        try:
            affinity = len(os.sched_getaffinity(0))
        except:
            affinity = max(1, n_workers)
        if n_workers == -1:
            self.n_workers = affinity
        else:
            self.n_workers = min(n_workers, affinity)

        print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        n_episodes_per_process = n_episodes // self.n_workers
        remainder = n_episodes % self.n_workers

        f = lambda pi, env: traj_segment_function(pi, env, n_episodes_per_process, horizon, stochastic)
        f_rem = lambda pi, env: traj_segment_function(pi, env, n_episodes_per_process+1, horizon, stochastic)
        fun = [f] * (self.n_workers - remainder) + [f_rem] * remainder
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], make_env, make_pi, fun[i], seed + i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()


    def collect(self, actor_weights):
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', actor_weights))

        for e in self.events:
            e.set()

        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.append(samples)

        return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):
        '''
        {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
         "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
         "ep_rets": ep_rets, "ep_lens": ep_lens, "mask": mask}
         '''
        np_fields = ['ob', 'rew', 'vpred', 'new', 'ac', 'prevac', 'mask']
        list_fields = ['ep_rets', 'ep_lens', 'policy_cum_time', 'policy_call_counter', 'total_time', 'env_time']

        new_dict = list(zip(np_fields, map(lambda f: sample_batches[0][f], np_fields))) + \
                   list(zip(list_fields,map(lambda f: sample_batches[0][f], list_fields))) + \
                   [('nextvpred', sample_batches[-1]['nextvpred'])]
        new_dict = dict(new_dict)

        for batch in sample_batches[1:]:
            for f in np_fields:
                new_dict[f] = np.concatenate((new_dict[f], batch[f]))
            for f in list_fields:
                new_dict[f].extend(batch[f])

        return new_dict


    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()

        for w in self.workers:
            w.join()
