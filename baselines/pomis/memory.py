"""
This class implements the memory required for multiple importance sampling.
Practically, it is a buffer of trajectories and policies.
"""

import numpy as np

class Memory():

    def __init__(self, capacity=10, batch_size=100, horizon=500, ob_space=None,
                 ac_space=None, strategy='fifo'):
        self.capacity = capacity
        self.batch_size = batch_size
        self.horizon = horizon
        self.ob_shape = list(ob_space.shape)
        self.ac_shape = list(ac_space.shape)
        self.strategy = strategy
        # Init the trajectory buffer
        self.trajectory_buffer = {
            'ob': None,
            'ac': None,
            'disc_rew': None,
            'rew': None,
            'new': None,
            'mask': None
        }

    def unflatten_batch_dict(self, batch):
        return {k: np.reshape(batch[k], [1, self.batch_size, self.horizon] + list(np.array(batch[k]).shape[1:])) for k in self.trajectory_buffer.keys()}

    def flatten_batch_dict(self, batch):
        return {k: np.reshape(v, [None] + list(v.shape[3:])) for k,v in batch.items()}

    def trim_batch(self):
        if self.strategy == 'fifo':
            for k, v in self.trajectory_buffer.items():
                if v is not None and v.shape[0] == self.capacity:
                    # We remove the first one since they are inserted in order
                    self.trajectory_buffer[k] = np.delete(v, 0, axis=0)
        else:
            raise Exception('Trimming strategy not recognized.')

    def add_trajectory_batch(self, batch):
        # First, trim the batch if the capacity is reached
        self.trim_batch()
        # Update with the new batch
        batch = self.unflatten_batch_dict(batch)
        for k, v in self.trajectory_buffer.items():
            if v is None:
                self.trajectory_buffer[k] = batch[k]
            else:
                self.trajectory_buffer[k] = np.concatenate((self.trajectory_buffer[k], batch[k]), axis=0)
            print(k, self.trajectory_buffer[k].shape)

    def get_trajectories(self):
        return self.flatten_batch_dict(self.trajectory_buffer)
