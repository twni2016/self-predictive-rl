from typing import Optional

import dm_env
from dm_env import specs

import numpy as np

import jax
import jax.numpy as jnp

from replearn import rollout


class MountainCarPolicy(rollout.EpsilonGreedyPolicy):
    def preferences(self, observations):
        # prefer action along the current velocity
        return jax.lax.cond(
            observations[1] < 0,
            lambda _: jnp.array([1.0, 0.0, 0.0]),
            lambda _: jnp.array([0.0, 0.0, 1.0]),
            None,
        )


class MountainCar(dm_env.Environment):
    """Implementation of the Mountain Car domain.

    Moore, Andrew William. "Efficient memory-based learning for robot control." (1990).

    Default parameters use values presented in Example 10.1 by Sutton & Barto (2018):
    Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press,
    2018.

    """

    def __init__(
        self,
        seed: Optional[int] = None,
        min_pos: float = -1.2,
        max_pos: float = 0.6,
        min_init_pos: float = -0.6,
        max_init_pos: float = -0.4,
        max_speed: float = 0.07,
        goal_pos: float = 0.5,
        force: float = 0.001,
        gravity: float = 0.0025,
    ):
        self._min_pos = min_pos
        self._max_pos = max_pos
        self._min_init_pos = min_init_pos
        self._max_init_pos = max_init_pos
        self._max_speed = max_speed
        self._goal_pos = goal_pos
        self._force = force
        self._gravity = gravity

        self._rng = np.random.default_rng(seed)
        self._position = 0.0
        self._velocity = 0.0

    def _observation(self):
        return np.array([self._position, self._velocity], np.float32)

    def reset(self):
        self._position = self._rng.uniform(self._min_init_pos, self._max_init_pos)
        self._velocity = 0.0
        return dm_env.restart(self._observation())

    def step(self, action):
        """Step the environment

        :param action: 0, 1, 2 correspond to actions left, idle, right, respectively.
        :return: the next timestep
        """
        next_vel = (
            self._velocity
            + self._force * (action - 1)
            - self._gravity * np.cos(self._position * 3)
        )
        self._velocity = np.clip(next_vel, -self._max_speed, self._max_speed)

        self._position = np.clip(
            self._position + next_vel, self._min_pos, self._max_pos
        )

        reward = -1
        obs = self._observation()

        if self._position >= self._goal_pos:
            return dm_env.termination(reward=0.0, observation=obs)

        return dm_env.transition(reward=reward, observation=obs)

    def observation_spec(self):
        return specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=[self._min_pos, -self._max_speed],
            maximum=[self._max_pos, self._max_speed],
        )

    def action_spec(self):
        """Actions  0, 1, 2 correspond to actions left, idle, right, respectively."""
        return specs.DiscreteArray(3, name="action")
