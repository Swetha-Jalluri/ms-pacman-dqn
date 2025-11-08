import numpy as np
from collections import deque, namedtuple
Transition = namedtuple("Transition", "obs action reward next_obs done")

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, dtype=np.uint8):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
        )
