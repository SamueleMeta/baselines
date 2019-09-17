from multiprocessing import Process, Queue, Event
import os
import baselines.common.tf_util as U
import time
import sys
from mpi4py import MPI
from baselines.common import set_global_seeds as set_all_seeds
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def traj_segment_function(pi, env, n_episodes, horizon, stochastic, n_samples=None):
    '''
    Collects trajectories
    '''

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    max_samples = n_samples if n_samples is not None else horizon * n_episodes

    # Initialize history arrays
    obs = np.array([ob for _ in range(max_samples)])
    rews = np.zeros(max_samples, 'float32')
    vpreds = np.zeros(max_samples, 'float32')
    news = np.zeros(max_samples, 'int32')
    acs = np.array([ac for _ in range(max_samples)])
    prevacs = acs.copy()
    mask = np.ones(max_samples, 'float32')

    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if (n_samples is None and i == n_episodes) or (n_samples is not None and t == n_samples):
           print(t, len(ep_lens), np.sum(news)) 
           return {"ob" : obs[:t], "rew" : rews[:t], "vpred" : vpreds[:t], "new" : news[:t],
                    "ac" : acs[:t], "prevac" : prevacs[:t], "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "mask" : mask[:t]}

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1
        if new or j == horizon or (n_samples is not None and t+1 == n_samples):
            new = True
            env.done = True

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

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

        pi = self.make_pi('pi%s' % os.getpid(), env.observation_space, env.action_space)
        #print('Worker %s - Running with seed %s' % (os.getpid(), workerseed))

        while True:
            self.event.wait()
            self.event.clear()
            command, weights = self.input.get()
            if command == 'collect':
                #print('Worker %s - Collecting... %s' % (os.getpid(), weights))
                pi.set_parameter(weights)
                samples = self.traj_segment_generator(pi, env)
                #print('Worker %s - Collected: %s' % (os.getpid(), samples['ob'].shape))
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                #print('Worker %s - Exiting...' % os.getpid())
                env.close()
                sess.close()
                break

class ParallelSampler(object):

    def __init__(self, make_pi, make_env, n_episodes, horizon, stochastic, n_samples=None, n_workers=-1, seed=0):
        try:
            affinity = len(os.sched_getaffinity(0))
            if n_workers == -1:
                self.n_workers = affinity
            else:
                self.n_workers = min(n_workers, affinity)
        except:
            self.n_workers = max(1, n_workers)

        #print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        if n_samples is None:
            n_episodes_per_process = n_episodes // self.n_workers
            remainder = n_episodes % self.n_workers

            f = lambda pi, env: traj_segment_function(pi, env, n_episodes_per_process, horizon, stochastic)
            f_rem = lambda pi, env: traj_segment_function(pi, env, n_episodes_per_process+1, horizon, stochastic)
            fun = [f] * (self.n_workers - remainder) + [f_rem] * remainder
            self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], make_env, make_pi, fun[i], seed+i*100) for i in range(self.n_workers)]
        else:
            n_samples_per_process = n_samples // self.n_workers
            remainder = n_samples % self.n_workers

            print("%s %s %s" % (n_samples_per_process, remainder, self.n_workers))

            f = lambda pi, env: traj_segment_function(pi, env, None, horizon, stochastic, n_samples=n_samples_per_process)
            f_rem = lambda pi, env: traj_segment_function(pi, env, None, horizon, stochastic, n_samples=n_samples_per_process+1)
            fun = [f] * (self.n_workers - remainder) + [f_rem] * remainder
            self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], make_env, make_pi, fun[i], seed + i * 100) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()


    def collect(self, actor_weights):
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', actor_weights))

        for e in self.events:
            e.set()

        sample_batches = [self.output_queue.get() for _ in range(self.n_workers)]
        sample_batches = sorted(sample_batches, key=lambda x: x[0])
        _, sample_batches = zip(*sample_batches)

        return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):
        '''
        {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
         "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
         "ep_rets": ep_rets, "ep_lens": ep_lens, "mask": mask}
         '''

        n_episodes_eff = 0
        n_samples_eff = 0
        horizon_eff = 0

        for batch in sample_batches:
            ep_lens = batch['ep_lens']
            n_episodes_eff += len(ep_lens)
            n_samples_eff += sum(ep_lens)
            horizon_eff = max(horizon_eff, max(ep_lens))

        np_fields = ['ob', 'rew', 'vpred', 'new', 'ac', 'prevac', 'mask']
        list_fields = ['ep_rets', 'ep_lens']

        new_dict = dict()
        for i, batch in enumerate(sample_batches):
            for f in np_fields:
                if i == 0:
                    new_dict[f] = self._pad(batch[f], batch['ep_lens'], horizon_eff, pad_value= 0 if f == 'mask' else None)
                else:
                    new_dict[f] = np.concatenate((new_dict[f], self._pad(batch[f], batch['ep_lens'], horizon_eff, pad_value= 0 if f == 'mask' else None)))
            for f in list_fields:
                if i == 0:
                    new_dict[f] = batch[f]
                else:
                    new_dict[f].extend(batch[f])

        return new_dict, horizon_eff, n_episodes_eff, n_samples_eff

    def _pad(self, field, ep_lens, horizon_eff, pad_value=None):

        len_cumsum = np.cumsum(ep_lens)

        starts = np.concatenate(([0], len_cumsum[:-1]))
        stops = len_cumsum

        # Strange hack
        shape_list = list(field.shape)
        shape_list[0] = len(ep_lens) * horizon_eff
        shape_tuple = tuple(shape_list)

        new_field = np.zeros(shape_tuple)

        for i, (start, stop) in enumerate(zip(starts, stops)):
            new_field[i*horizon_eff:i*horizon_eff+ep_lens[i]] = field[start:stop]
            if pad_value is None:
                new_field[i*horizon_eff+ep_lens[i]:(i + 1)*horizon_eff] = field[stop-1]
            else:
                new_field[i*horizon_eff+ep_lens[i]:(i + 1) * horizon_eff] = pad_value

        return new_field


    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()

        for w in self.workers:
            w.join()
