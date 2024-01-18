from typing import Any, NamedTuple, Optional

import dm_env
from dm_env import specs

import numpy as np

import jax
import jax.numpy as jnp

from replearn import rollout


class LoadUnloadState(NamedTuple):
    position: int
    loaded: bool


class LoadUnloadPolicyState(NamedTuple):
    key: Any
    last_action: int


class LoadUnloadPolicy:
    def __init__(self, key, switch_prob=0.2):
        self._state = LoadUnloadPolicyState(key, 0)
        self._swtich_prob = switch_prob

        def _sample_and_next_state(state, observations):
            key, a_key = jax.random.split(state.key)
            action = self.sample(
                LoadUnloadPolicyState(a_key, state.last_action), observations
            )
            return LoadUnloadPolicyState(key, action), action

        self.sample_and_next_state = jax.jit(_sample_and_next_state)

    def sample(self, state, observations):
        def _maybe_switch_action():
            return jax.lax.cond(
                jax.random.uniform(state.key) < self._swtich_prob,
                lambda: (state.last_action + 1) % 2,
                lambda: state.last_action,
            )

        # if at unload go right, if at load go left, else keep previous action with random switch
        return jax.lax.switch(
            observations[0], [lambda: 1, _maybe_switch_action, lambda: 0]
        )

    def stateful_sample(self, observations):
        self._state, action = self.sample_and_next_state(self._state, observations)
        return action

    def set_rng_key(self, key):
        self._state = LoadUnloadPolicyState(key, self._state.last_action)


class LoadUnload(dm_env.Environment):
    def __init__(self, seed: Optional[int] = None, load_position: Optional[int] = 6):
        self._state = LoadUnloadState(0, False)
        self._load_position = load_position
        self._rng = np.random.default_rng(seed)

    def _observation(self):
        if self._state.position == 0:
            obs = 0
        elif self._state.position == self._load_position:
            obs = 2
        else:
            obs = 1
        return np.array([obs], np.int32)

    def reset(self):
        new_pos = self._rng.integers(self._load_position, size=1)[0]
        self._state = LoadUnloadState(
            new_pos,
            new_pos == self._load_position,
        )
        return dm_env.restart(self._observation())

    def step(self, action):
        pos, loaded = self._state
        if action == 0:
            new_pos = max((pos - 1, 0))
        elif action == 1:
            new_pos = min((pos + 1, self._load_position))
        else:
            raise ValueError(f"Unrecognized action '{action}'")

        reward = -1
        if new_pos == 0:
            if loaded:
                reward = 100
            loaded = False
        elif new_pos == self._load_position:
            loaded = True

        self._state = LoadUnloadState(new_pos, loaded)
        obs = self._observation()

        return dm_env.transition(reward=reward, observation=obs)

    def observation_spec(self):
        return specs.DiscreteArray(3, name="observation")

    def action_spec(self):
        """Actions  0, 1 correspond to actions left, right, respectively."""
        return specs.DiscreteArray(2, name="action")
