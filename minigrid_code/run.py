import numpy as np
from agent import Agent
from r2d2replaybuffer import r2d2_ReplayMemory
import torch
import gymnasium as gym
import time
import logger


def run_exp(args):
    ## Create env
    assert "MiniGrid-" in args["env_name"]
    env = gym.make(args["env_name"])
    test_env = gym.make(args["env_name"])

    obs_dim = np.prod(env.observation_space["image"].shape)  # only use image obs
    act_dim = env.action_space.n  # discrete action
    logger.log(
        env.observation_space["image"],
        f"obs_dim={obs_dim} act_dim={act_dim} max_steps={env.max_steps}",
    )

    ## Initialize agent and buffer
    agent = Agent(env, args)

    memory = r2d2_ReplayMemory(args["replay_size"], obs_dim, act_dim, args)

    ## Training
    seed = args["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    test_env.reset(seed=seed + 1)
    memory.reset(seed)

    total_numsteps = 0
    while total_numsteps <= args["num_steps"]:
        hidden_p = agent.get_initial_hidden()
        action = -1  # placeholder
        reward = 0
        state = env.reset()[0]["image"].astype(np.float32).reshape(-1)

        ep_hiddens = [hidden_p]  # z[-1]
        ep_actions = [action]  # a[-1]
        ep_rewards = [reward]  # r[-1]
        ep_states = [state]  # o[0]

        while True:
            if total_numsteps % args["logging_freq"] == 0:
                if total_numsteps > 0:  # except the first evaluation
                    FPS = running_metrics["length"] / (time.time() - time_now)
                    # average the metrics
                    running_metrics = {
                        k: v / k_episode for k, v in running_metrics.items()
                    }
                    running_losses = {
                        k: v / k_updates for k, v in running_losses.items()
                    }
                log_and_test(
                    test_env,
                    agent,
                    total_numsteps,
                    running_metrics if total_numsteps > 0 else None,
                    running_losses if total_numsteps > 0 else None,
                    FPS if total_numsteps > 0 else None,
                )
                ## running metrics
                k_episode = 0  # num of env episodes
                k_updates = 0  # num of agent updates
                running_metrics = {
                    k: 0.0
                    for k in [
                        "return",
                        "length",
                        "success",
                    ]
                }
                running_losses = {}
                time_now = time.time()

            if total_numsteps < args["random_actions_until"]:  # never used
                action = env.action_space.sample()
            else:
                action, hidden_p = agent.select_action(
                    state,
                    action,
                    reward,
                    hidden_p,
                    EPS_up=True,
                    evaluate=False,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)  # Step
            state = next_state["image"].astype(np.float32).reshape(-1)

            ep_hiddens.append(hidden_p)  # z[t]
            ep_actions.append(action)  # a[t]
            ep_rewards.append(reward)  # r[t]
            ep_states.append(state)  # o[t+1]

            running_metrics["return"] += reward
            running_metrics["length"] += 1

            if (
                len(memory) > args["batch_size"]
                and total_numsteps % args["rl_update_every_n_steps"] == 0
            ):
                losses = agent.update_parameters(
                    memory, args["batch_size"], args["rl_updates_per_step"]
                )
                k_updates += 1
                if running_losses == {}:
                    running_losses = losses
                else:
                    running_losses = {
                        k: running_losses[k] + v for k, v in losses.items()
                    }

            total_numsteps += 1

            if terminated or truncated:
                break

        # Append transition to memory
        memory.push(ep_states, ep_actions, ep_rewards, ep_hiddens)

        k_episode += 1
        running_metrics["success"] += int(reward > 0.0)  # terminal reward


def log_and_test(
    env,
    agent,
    total_numsteps,
    running_metrics,
    running_losses,
    FPS,
):
    logger.record_step("env_steps", total_numsteps)
    if total_numsteps > 0:
        for k, v in running_metrics.items():
            logger.record_tabular("train/" + k, v)
        for k, v in running_losses.items():
            logger.record_tabular(k, v)
        logger.record_tabular("FPS", FPS)

    metrics = {
        k: 0.0
        for k in [
            "return",
            "length",
            "success",
        ]
    }
    episodes = 10
    for _ in range(episodes):
        hidden_p = agent.get_initial_hidden()
        action = -1  # placeholder
        reward = 0
        state = env.reset()[0]["image"].astype(np.float32).reshape(-1)

        while True:
            action, hidden_p = agent.select_action(
                state, action, reward, hidden_p, EPS_up=False, evaluate=True
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            metrics["return"] += reward
            metrics["length"] += 1

            state = next_state["image"].astype(np.float32).reshape(-1)

            if terminated or truncated:
                break

        metrics["success"] += int(reward > 0.0)

    metrics = {k: metrics[k] / episodes for k in metrics.keys()}
    for k, v in metrics.items():
        logger.record_tabular(k, v)
    logger.dump_tabular()
