from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

import numpy as np
from copy import deepcopy
import haiku as hk
import optax


class Parameters(NamedTuple):
    encoder: Any
    transition: Any


def create_latent_encoder(latent_size):
    def encode_latent_state(state):
        return hk.Linear(
            latent_size,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
        )(state)

    return hk.transform(encode_latent_state)


def apply_transition(trans_matrix, latent_state, action):
    state_action = jnp.concatenate((latent_state, action))
    return jnp.dot(trans_matrix.T, state_action)


def train(
    key,
    optimizer,
    encoder,
    states,
    actions,
    next_states,
    use_stop_gradient: str,
    num_steps,
    log_n_steps,
):
    def solve_transition_params(encoder_params, target_encoder_params):
        z_t = encoder.apply(encoder_params, None, states)
        za_t = jnp.concatenate((z_t, actions), axis=1)
        z_tp1 = encoder.apply(target_encoder_params, None, next_states)
        opt_trans_params, *_ = jnp.linalg.lstsq(za_t, z_tp1)
        return opt_trans_params

    def loss(encoder_params, target_encoder_params, s_t, a_t, s_tp1):
        batch_encoder_apply = jax.vmap(encoder.apply, (None, None, 0))
        z_t = batch_encoder_apply(encoder_params, None, s_t)

        if use_stop_gradient == "Detached":
            z_tp1 = batch_encoder_apply(
                jax.lax.stop_gradient(encoder_params), None, s_tp1
            )
            opt_trans_params = jax.lax.stop_gradient(
                solve_transition_params(encoder_params, encoder_params)
            )
        elif use_stop_gradient == "Online":
            z_tp1 = batch_encoder_apply(encoder_params, None, s_tp1)
            opt_trans_params = jax.lax.stop_gradient(
                solve_transition_params(encoder_params, encoder_params)
            )
        else:  # EMA
            z_tp1 = batch_encoder_apply(target_encoder_params, None, s_tp1)
            opt_trans_params = jax.lax.stop_gradient(
                solve_transition_params(encoder_params, target_encoder_params)
            )

        estimated_z_tp1 = jax.vmap(apply_transition, (None, 0, 0))(
            opt_trans_params, z_t, a_t
        )
        error = estimated_z_tp1 - z_tp1

        return 0.5 * jnp.mean(jnp.sum(error**2, axis=-1))

    @jax.jit
    def step(params, target_params, opt_state, s_t, a_t, s_tp1):
        loss_value, grads = jax.value_and_grad(loss)(
            params, target_params, s_t, a_t, s_tp1
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_params = target_update(target_params, params)

        return params, target_params, opt_state, loss_value

    def target_update(
        target_params,
        new_params,
        tau: float = 0.005,
    ):
        return jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), new_params, target_params
        )

    assert use_stop_gradient in ["Online", "Detached", "EMA"]

    key, init_key = jax.random.split(key)
    encoder_params = encoder.init(init_key, states[0])
    target_encoder_params = deepcopy(encoder_params)
    opt_state = optimizer.init(encoder_params)

    loss_value = jax.jit(loss)(
        encoder_params, target_encoder_params, states, actions, next_states
    )

    logs = []
    for i in range(num_steps):
        if i % log_n_steps == 0:
            params = encoder_params["linear"]["w"]
            logs.append(
                {
                    "step": i,
                    "loss": float(loss_value),
                    "params": np.array(params),
                }
            )

        encoder_params, target_encoder_params, opt_state, loss_value = step(
            params=encoder_params,
            target_params=target_encoder_params,
            opt_state=opt_state,
            s_t=states,
            a_t=actions,
            s_tp1=next_states,
        )

    params = encoder_params["linear"]["w"]
    logs.append(
        {
            "step": num_steps,
            "loss": float(loss_value),
            "params": np.array(params),
        }
    )

    return logs
