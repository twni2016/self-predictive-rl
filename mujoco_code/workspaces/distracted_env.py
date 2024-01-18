import gym
import numpy as np


class DistractedWrapper(gym.Wrapper):
    def __init__(self, env, distraction: int, scale: float = 1.0) -> None:
        super().__init__(env)
        assert distraction > 0
        self.d = distraction
        self.scale = scale
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.reset().shape, dtype=np.float32
        )

    def _get_distract_obs(self):
        return self.scale * np.random.normal(size=(self.d,))

    def reset(self):
        obs = self.env.reset()
        return np.concatenate([obs, self._get_distract_obs()])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return np.concatenate([obs, self._get_distract_obs()]), rew, done, info
