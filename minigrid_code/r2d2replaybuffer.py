import random
import torch
import torch.nn.functional as F
import numpy as np


class r2d2_ReplayMemory:
    def __init__(self, capacity, obs_dim, act_dim, args):
        self.capacity = capacity
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = args["gamma"]
        self.burn_in_len = args["burn_in_len"]  # H = 50
        self.learning_obs_len = args["learning_obs_len"]  # L = 10
        self.forward_len = args["forward_len"]  # N = 5
        self.AIS_state_size = args["AIS_state_size"]
        self.batch_size = args["batch_size"]

        self.buffer_hidden = (
            np.zeros([self.capacity, self.AIS_state_size], dtype=np.float32),
            np.zeros([self.capacity, self.AIS_state_size], dtype=np.float32),
        )
        self.buffer_burn_in_len = np.zeros([self.capacity], dtype=np.int32)
        self.buffer_burn_in_history = np.zeros(
            [self.capacity, self.burn_in_len, self.obs_dim + self.act_dim + 1],
            dtype=np.float32,
        )

        self.buffer_learning_len = np.zeros([self.capacity], dtype=np.int32)
        self.buffer_learning_history = np.zeros(
            [
                self.capacity,
                self.learning_obs_len + self.forward_len,
                self.obs_dim + self.act_dim + 1,
            ],
            dtype=np.float32,
        )

        self.buffer_learn_forward_len = np.zeros([self.capacity], dtype=np.int32)
        self.buffer_forward_idx = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.int32
        )
        self.buffer_current_act = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.int32
        )
        self.buffer_next_obs = np.zeros(
            [self.capacity, self.learning_obs_len, self.obs_dim],
            dtype=np.float32,
        )
        self.buffer_rewards = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.float32
        )
        self.buffer_model_target_rewards = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.float32
        )

        self.buffer_final_flag = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.int32
        )
        self.buffer_model_final_flag = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.int32
        )
        self.buffer_gammas = np.zeros(
            [self.capacity, self.learning_obs_len], dtype=np.float32
        )

        self.position_r2d2 = 0

        self.full = False

    def save_buffer(self, dir, seed):
        import os

        path = os.path.join(dir, "Seed_" + str(seed) + "_replaybuffer.pt")

        torch.save(
            {
                "buffer_burn_in_history": self.buffer_burn_in_history,
                "buffer_learning_history": self.buffer_learning_history,
                "buffer_current_act": self.buffer_current_act,
                "buffer_next_obs": self.buffer_next_obs,
                "buffer_rewards": self.buffer_rewards,
                "buffer_model_target_rewards": self.buffer_model_target_rewards,
                "buffer_burn_in_len": self.buffer_burn_in_len,
                "buffer_forward_idx": self.buffer_forward_idx,
                "buffer_learning_len": self.buffer_learning_len,
                "buffer_hidden_1": self.buffer_hidden[0],
                "buffer_hidden_2": self.buffer_hidden[1],
                "buffer_final_flag": self.buffer_final_flag,
                "buffer_model_final_flag": self.buffer_model_final_flag,
                "buffer_gammas": self.buffer_gammas,
            },
            path,
        )

    def reset(self, seed):
        random.seed(seed)

        self.position_r2d2 = 0
        self.full = False

    def push(self, ep_states, ep_actions, ep_rewards, ep_hiddens):
        """
        add an entire episode to the buffer
        """
        ep_states = np.array(ep_states)  # (T+1, O)
        ep_actions = np.array(ep_actions)  # (T+1,)
        ep_rewards = np.array(ep_rewards)  # (T+1,)
        ep_hiddens = np.array(
            [
                [
                    ep_hidden[0].cpu().numpy().flatten(),
                    ep_hidden[1].cpu().numpy().flatten(),
                ]
                for ep_hidden in ep_hiddens
            ]
        )  # (T+1, 2, Z) assume LSTM

        # Prepare raw data
        ls_prev_rewards = ep_rewards[:-1]  # (T)
        ls_curr_rewards = ep_rewards[1:]  # (T)

        ls_curr_actions = F.one_hot(
            torch.LongTensor(ep_actions[1:]), num_classes=self.act_dim
        )
        ls_curr_actions = ls_curr_actions.numpy().astype(np.int32)  # (T, A)
        ls_prev_actions = np.concatenate(
            [np.zeros((1, self.act_dim), dtype=np.int32), ls_curr_actions[:-1]], axis=0
        )  # (T, A)

        ls_curr_obs = ep_states[:-1]  # (T, O)
        ls_next_obs = ep_states[1:]  # (T, O)
        ls_hiddens = ep_hiddens[:-1]  # (T, 2, Z)
        T = len(ls_curr_obs)

        ### Prepare burn-in history: early items are shorter than burn_in_len
        hidden_list = [
            ls_hiddens[max(0, x - self.burn_in_len)]
            for x in range(0, T, self.learning_obs_len)
        ]
        burn_in_act_list = [
            ls_prev_actions[max(0, x - self.burn_in_len) : x]
            for x in range(0, T, self.learning_obs_len)
        ]
        burn_in_r_list = [
            ls_prev_rewards[max(0, x - self.burn_in_len) : x]
            for x in range(0, T, self.learning_obs_len)
        ]
        burn_in_obs_list = [
            ls_curr_obs[max(0, x - self.burn_in_len) : x]
            for x in range(0, T, self.learning_obs_len)
        ]

        ### Prepare learning data: late items are shorter than self.learning_obs_len + self.forward_len
        ### They do not include the terminal tuple of action, reward, next_obs
        learning_act_list = [
            ls_prev_actions[x : x + self.learning_obs_len + self.forward_len]
            for x in range(0, T, self.learning_obs_len)
        ]
        learning_r_list = [
            ls_prev_rewards[x : x + self.learning_obs_len + self.forward_len]
            for x in range(0, T, self.learning_obs_len)
        ]
        learning_obs_list = [
            ls_curr_obs[x : x + self.learning_obs_len + self.forward_len]
            for x in range(0, T, self.learning_obs_len)
        ]

        ### Prepare TD and AIS data (this include terminal action, obs, and reward)
        current_act_list = [
            ls_curr_actions[x : x + self.learning_obs_len]
            for x in range(0, T, self.learning_obs_len)
        ]
        next_obs_list = [
            ls_next_obs[x : x + self.learning_obs_len]
            for x in range(0, T, self.learning_obs_len)
        ]  # for one-step prediction, instead of forward_len-step prediction in prior work
        ep_rewards_list = [
            ls_curr_rewards[x : x + self.learning_obs_len]
            for x in range(0, T, self.learning_obs_len)
        ]  # for one-step prediction, instead of forward_len-step prediction in prior work

        ep_rewards_ = ls_curr_rewards[:-1]
        discounted_sum = [
            [
                sum_rewards(ep_rewards_[x + y : x + y + self.forward_len], self.gamma)
                if x + y != len(ep_rewards_)
                else ls_curr_rewards[x + y]
                for y in range(0, min(self.learning_obs_len, T - x))
            ]
            for x in range(0, T, self.learning_obs_len)
        ]

        ### Store into the buffer
        for i in range(len(hidden_list)):
            # store burn-in
            self.buffer_burn_in_len[self.position_r2d2] = len(burn_in_obs_list[i])
            self.buffer_hidden[0][self.position_r2d2, :] = hidden_list[i][0]
            self.buffer_hidden[1][self.position_r2d2, :] = hidden_list[i][1]
            if len(burn_in_obs_list[i]) != 0:
                self.buffer_burn_in_history[
                    self.position_r2d2, : len(burn_in_act_list[i]), :
                ] = np.concatenate(
                    (
                        burn_in_obs_list[i],
                        burn_in_act_list[i],
                        burn_in_r_list[i].reshape(-1, 1),
                    ),
                    axis=-1,
                )

            # store learn data
            self.buffer_learn_forward_len[self.position_r2d2] = len(
                learning_act_list[i]
            )
            self.buffer_learning_history[
                self.position_r2d2, : len(learning_act_list[i]), :
            ] = np.concatenate(
                (
                    learning_obs_list[i],
                    learning_act_list[i],
                    learning_r_list[i].reshape(-1, 1),
                ),
                axis=-1,
            )

            # store TD and AIS data
            self.buffer_current_act[
                self.position_r2d2, : len(current_act_list[i])
            ] = np.argmax(current_act_list[i], axis=-1)
            self.buffer_next_obs[
                self.position_r2d2, : len(next_obs_list[i]), :
            ] = next_obs_list[i]
            self.buffer_model_target_rewards[
                self.position_r2d2, : len(ep_rewards_list[i])
            ] = ep_rewards_list[i]

            self.buffer_rewards[
                self.position_r2d2, : len(discounted_sum[i])
            ] = np.array(discounted_sum[i])
            self.buffer_learning_len[self.position_r2d2] = len(discounted_sum[i])
            self.buffer_forward_idx[
                self.position_r2d2, : len(discounted_sum[i])
            ] = np.array(
                [
                    min(j + self.forward_len, len(learning_obs_list[i]) - 1)
                    for j in range(len(discounted_sum[i]))
                ]
            )

            # NOTE: assume all dones are terminated, which is okay in minigrid tasks
            # where the timeout reward is exact 0.0,
            # and the training code is hard to adapt to timeout scenarios, as it
            self.buffer_final_flag[
                self.position_r2d2, : len(discounted_sum[i])
            ] = np.array(
                [
                    int(i * self.learning_obs_len + j < T - 1)
                    for j in range(len(discounted_sum[i]))
                ]
            )
            self.buffer_model_final_flag[
                self.position_r2d2, : len(discounted_sum[i])
            ] = np.array(
                [
                    int(i * self.learning_obs_len + j <= T - 1)
                    for j in range(len(discounted_sum[i]))
                ]
            )  # this flag includes terminal step, used for reward prediction only

            self.buffer_gammas[self.position_r2d2, : len(discounted_sum[i])] = np.array(
                [
                    self.gamma
                    ** (min(j + self.forward_len, len(learning_obs_list[i]) - 1) - j)
                    for j in range(len(discounted_sum[i]))
                ]
            )

            if self.full is False and self.position_r2d2 + 1 == self.capacity:
                self.full = True
            self.position_r2d2 = (self.position_r2d2 + 1) % self.capacity

    def sample(self, batch_size):
        tmp = self.position_r2d2
        if self.full:
            tmp = self.capacity
        idx = np.random.choice(tmp, batch_size, replace=False)

        batch_burn_in_hist = self.buffer_burn_in_history[idx, :, :]
        batch_learn_hist = self.buffer_learning_history[idx, :, :]
        batch_rewards = self.buffer_rewards[idx, :]
        batch_burn_in_len = self.buffer_burn_in_len[idx]
        batch_forward_idx = self.buffer_forward_idx[idx, :]
        batch_final_flag = self.buffer_final_flag[idx, :]
        batch_learn_len = self.buffer_learning_len[idx]
        batch_hidden = (self.buffer_hidden[0][idx], self.buffer_hidden[1][idx])
        batch_current_act = self.buffer_current_act[idx, :]
        batch_learn_forward_len = self.buffer_learn_forward_len[idx]
        batch_next_obs = self.buffer_next_obs[idx]
        batch_model_target_reward = self.buffer_model_target_rewards[idx]
        batch_model_final_flag = self.buffer_model_final_flag[idx, :]
        batch_gammas = self.buffer_gammas[idx]

        return (
            batch_burn_in_hist,
            batch_learn_hist,
            batch_rewards,
            batch_learn_len,
            batch_forward_idx,
            batch_final_flag,
            batch_current_act,
            batch_hidden,
            batch_burn_in_len,
            batch_learn_forward_len,
            batch_next_obs,
            batch_model_target_reward,
            batch_model_final_flag,
            batch_gammas,
        )

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.position_r2d2


def sum_rewards(reward_list, gamma):
    ls = [reward_list[i] * gamma**i for i in range(0, len(reward_list))]
    return sum(ls)
