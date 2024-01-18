from typing import Optional, Protocol, Sequence, Tuple
import functools

import numpy as np

import dm_env
from dm_env import specs

import chex
import jax
import rlax
from rlax._src import distributions


class Policy(Protocol):
    def sample(self, key, observations):
        raise NotImplementedError

    def stateful_sample(self, observations):
        raise NotImplementedError

    def set_rng_key(self, key):
        raise NotImplementedError


class DiscretePolicy(Policy, Protocol):
    action_distribution: distributions.DiscreteDistribution

    def probs(self, observations):
        raise NotImplementedError


class EpsilonGreedyPolicy(DiscretePolicy):
    def __init__(self, key, epsilon):
        self._key = key
        self.action_distribution = rlax.epsilon_greedy(epsilon)

        def _sample_and_split(key, observations):
            key, a_key = jax.random.split(key)
            return key, self.sample(a_key, observations)

        self.sample_and_split = jax.jit(_sample_and_split)

    def preferences(self, observations):
        raise NotImplementedError

    def sample(self, key, observations):
        return self.action_distribution.sample(key, self.preferences(observations))

    def probs(self, observations):
        return self.action_distribution.probs(self.preferences(observations))

    def set_rng_key(self, key):
        self._key = key

    def stateful_sample(self, observations):
        self._key, action = self.sample_and_split(self._key, observations)
        return action


def generate_trajectory(
    key: chex.PRNGKey,
    env: dm_env.Environment,
    policy: Policy,
    max_steps: Optional[int] = None,
) -> Tuple[Sequence[dm_env.TimeStep], Sequence]:
    t = 0
    timestep = env.reset()
    trajectory = [timestep]
    actions = []

    policy.set_rng_key(key)

    while (not timestep.last()) and (max_steps is None or t < max_steps):
        action = policy.stateful_sample(timestep.observation)
        timestep = env.step(action)

        t += 1
        trajectory.append(timestep)
        actions.append(action)

    return trajectory, actions


def traj_to_observation_array(trajectory):
    _, _, _, observations = zip(*trajectory)
    return np.array(observations)


def rollout_dataset(
    key,
    *,
    env_cls,
    policy,
    history_encoder,
    act_encoder,
    obs_encoder,
    max_traj_length: int,
    num_traj: Optional[int] = None,
    num_steps: Optional[int] = None
):
    if num_traj == num_steps:
        raise ValueError(
            (
                "Either `num_traj` or `num_steps` is required. Providing a value for both is not "
                "supported."
            )
        )

    env_seed, policy_seed = np.random.SeedSequence(key).spawn(2)
    policy_key = policy_seed.generate_state(2)

    env = env_cls(seed=env_seed.generate_state(1)[0])
    data = []

    traj_count = 0
    step_count = 0
    while (num_traj is None or traj_count < num_traj) and (
        num_steps is None or step_count < num_steps
    ):
        traj_len_limit = max_traj_length
        if num_steps is not None:
            traj_len_limit = min((traj_len_limit, num_steps - step_count))

        traj_key, policy_key = jax.random.split(policy_key)
        traj, actions = generate_trajectory(
            traj_key, env, policy, max_steps=traj_len_limit
        )

        actions = act_encoder.apply(np.array(actions))
        observations = traj_to_observation_array(traj)
        observations = obs_encoder.apply(observations)
        history = history_encoder.apply(observations, actions)

        data.append((history[:-1], actions, history[1:]))

        traj_count += 1
        step_count += data[-1][0].shape[0]

    states, actions, next_states = [np.concatenate(x) for x in zip(*data)]
    return states, actions, next_states
