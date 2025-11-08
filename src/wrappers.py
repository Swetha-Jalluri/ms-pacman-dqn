import cv2
import numpy as np
import gymnasium as gym
from collections import deque

class PreprocessObs(gym.ObservationWrapper):
    def __init__(self, env, resize=(84,84), gray=True):
        super().__init__(env)
        self.resize = resize
        self.gray = gray
        c = 1 if gray else 3
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, resize[0], resize[1]), dtype=np.uint8
        )
    def observation(self, obs):
        # obs: H x W x C (RGB)
        if self.gray:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.resize, interpolation=cv2.INTER_AREA)
        if self.gray:
            obs = np.expand_dims(obs, axis=2)  # H W 1
        obs = np.transpose(obs, (2,0,1))      # C H W
        return obs.astype(np.uint8)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c*k, h, w), dtype=np.uint8
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k): self.frames.append(obs)
        return self._get_obs(), info
    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), r, terminated, truncated, info
    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

class ClipReward(gym.RewardWrapper):
    def reward(self, reward):
        # {-1, 0, +1} clipping for stability
        return float(np.sign(reward))
