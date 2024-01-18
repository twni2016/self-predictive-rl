from collections import abc
import itertools
from typing import Any, NamedTuple, Protocol, TypeVar

import numpy as np

import jax
import jax.numpy as jnp

import dm_env

T = TypeVar("T", bound=Any, covariant=True)
PyTree = Any
Array = Any


class Encoder(Protocol[T]):
    def apply(self, observation: PyTree) -> T:
        raise NotImplementedError


def _as_iterable(bound, max_repeat=None):
    if not isinstance(bound, abc.Iterable):
        bound = itertools.repeat(bound, max_repeat)
    return bound


def uniform_centers(env: dm_env.Environment, centers_per_dim: int):
    obs_spec = env.observation_spec()
    assert len(obs_spec.shape) == 1, "Only rank 1 observations are supported."

    minimums = _as_iterable(obs_spec.minimum, obs_spec.shape[0])
    maximums = _as_iterable(obs_spec.maximum, obs_spec.shape[0])

    centers = np.meshgrid(
        *[
            np.linspace(lb, ub, centers_per_dim, endpoint=True)
            for lb, ub in zip(minimums, maximums)
        ]
    )
    centers = np.stack([x.flatten() for x in centers], axis=0)

    return centers


def normalized_scales(env: dm_env.Environment, scale: float):
    obs_spec = env.observation_spec()
    assert len(obs_spec.shape) == 1, "Only rank 1 observations are supported."

    minimums = list(_as_iterable(obs_spec.minimum, obs_spec.shape[0]))
    maximums = list(_as_iterable(obs_spec.maximum, obs_spec.shape[0]))
    span = np.subtract(maximums, minimums)

    return span * scale


class RBFEncoder(NamedTuple):
    """A feature encoding using radial basis functions."""

    centers: Array
    scales: Array
    normalized: bool

    def apply(self, inputs):
        diff = (inputs[..., None] - self.centers) / self.scales[..., None]
        neg_dist = -jnp.sum(diff**2, axis=-2)
        if self.normalized:
            return jax.nn.softmax(neg_dist)
        else:
            return jnp.exp(neg_dist)


class OneHot(NamedTuple):
    """A one-hot encoder."""

    dim: int

    def apply(self, inputs):
        if inputs.shape[-1] == 1:
            inputs = jnp.squeeze(inputs, -1)
        return jax.nn.one_hot(inputs, self.dim)


class TruncatedHistoryEncoder(NamedTuple):
    horizon: int

    def apply(self, observations, actions):
        observations = jnp.pad(observations, [(self.horizon - 1, 0), (0, 0)])
        actions = jnp.pad(actions, [(self.horizon - 1, 0), (0, 0)])
        stacked_obs = [
            self.index_to_history(i, observations, actions)
            for i in range(0, observations.shape[0] - self.horizon + 1)
        ]
        return jnp.stack(stacked_obs)

    def index_to_history(self, index, observations, actions):
        return jnp.concatenate(
            (
                observations[index : index + self.horizon].reshape(-1),
                actions[index : index + self.horizon - 1].reshape(-1),
            )
        )
