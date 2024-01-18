from models import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random
import logger


class Agent(object):
    def __init__(self, env, args):
        self.args = args
        self.device = torch.device("cuda" if args["cuda"] else "cpu")
        self.obs_dim = np.prod(env.observation_space["image"].shape)  # flatten
        self.act_dim = env.action_space.n
        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.target_update_interval = args["target_update_interval"]

        self.aux = args["aux"]
        assert self.aux in ["None", "ZP", "OP", "AIS", "AIS-P2"]
        self.AIS_state_size = args["AIS_state_size"]

        self.encoder = SeqEncoder(self.obs_dim, self.act_dim, self.AIS_state_size).to(
            self.device
        )
        self.encoder_target = SeqEncoder(
            self.obs_dim, self.act_dim, self.AIS_state_size
        ).to(self.device)
        hard_update(self.encoder_target, self.encoder)

        self.critic = QNetwork_discrete(
            self.AIS_state_size, self.act_dim, args["hidden_size"]
        ).to(device=self.device)
        self.critic_target = QNetwork_discrete(
            self.AIS_state_size, self.act_dim, args["hidden_size"]
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.aux in ["AIS", "AIS-P2"]:  # modular
            self.optim = Adam(
                self.critic.parameters(),
                lr=args["rl_lr"],
            )
        else:  # end-to-end
            self.optim = Adam(
                list(self.encoder.parameters()) + list(self.critic.parameters()),
                lr=args["rl_lr"],
            )

        if self.aux in ["AIS", "AIS-P2"]:
            self.model = AISModel(
                self.obs_dim if self.aux == "AIS" else self.AIS_state_size,
                self.act_dim,
                self.AIS_state_size,
            ).to(self.device)
            self.AIS_optim = Adam(
                list(self.encoder.parameters()) + list(self.model.parameters()),
                lr=args["aux_lr"],
            )
        elif self.aux == "None":  # model-free R2D2
            self.model = None
        else:
            self.model = LatentModel(
                self.obs_dim if self.aux == "OP" else self.AIS_state_size,
                self.act_dim,
                self.AIS_state_size,
            ).to(self.device)
            self.AIS_optim = Adam(
                self.model.parameters(),
                lr=args["aux_lr"],
            )

        logger.log(self.encoder, self.model, self.critic)

        self.aux_optim = args["aux_optim"]
        self.aux_coef = args["aux_coef"]
        assert self.aux_optim in ["None", "ema", "detach", "online"]
        assert self.aux_coef >= 0.0

        self.update_to_q = 0
        self.eps_greedy_parameters = {
            "EPS_START": args["EPS_start"],
            "EPS_END": args["EPS_end"],
            "EPS_DECAY": args["EPS_decay"],
        }
        self.env_steps = 0

        self.get_initial_hidden = lambda: self.encoder.get_initial_hidden(
            1, self.device
        )

    @torch.no_grad()
    def select_action(
        self, state, action, reward, hidden_p, EPS_up: bool, evaluate: bool
    ):
        action = convert_int_to_onehot(action, self.act_dim)
        reward = torch.Tensor([reward])
        state = torch.Tensor(state)
        rho_input = torch.cat((state, action, reward)).reshape(1, 1, -1).to(self.device)

        ais_z, hidden_p = self.encoder(
            rho_input,
            batch_size=1,
            hidden=hidden_p,
            device=self.device,
            batch_lengths=[],
            pack_sequence=False,
        )
        if evaluate is False and EPS_up:
            self.env_steps += 1

        if self.args["EPS_decay_type"] == "exponential":
            eps_threshold = self.eps_greedy_parameters["EPS_END"] + (
                self.eps_greedy_parameters["EPS_START"]
                - self.eps_greedy_parameters["EPS_END"]
            ) * math.exp(
                -1.0 * self.env_steps / self.eps_greedy_parameters["EPS_DECAY"]
            )
        elif self.args["EPS_decay_type"] == "linear":
            eps_threshold = self.eps_greedy_parameters["EPS_START"] + (
                self.eps_greedy_parameters["EPS_END"]
                - self.eps_greedy_parameters["EPS_START"]
            ) * (self.env_steps / self.eps_greedy_parameters["EPS_DECAY"])
            eps_threshold = max(eps_threshold, self.eps_greedy_parameters["EPS_END"])

        sample = random.random()
        if (sample < eps_threshold and evaluate is False) or (
            sample < self.args["test_epsilon"] and evaluate is True
        ):
            return random.randrange(self.act_dim), hidden_p

        qf = self.critic(ais_z)[0, 0]
        greedy_action = torch.argmax(qf).item()

        return greedy_action, hidden_p

    def update_parameters(self, memory, batch_size: int, updates: int):
        for _ in range(updates):
            self.update_to_q += 1
            metrics = self.single_update(memory, batch_size)

            if self.update_to_q % self.target_update_interval == 0:
                # We change the hard update to soft update
                soft_update(self.encoder_target, self.encoder, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)

        return metrics

    def report_rank(self, z_batch, metrics: dict):
        from torch.linalg import matrix_rank

        rank3 = matrix_rank(z_batch, atol=1e-3, rtol=1e-3)
        rank2 = matrix_rank(z_batch, atol=1e-2, rtol=1e-2)
        rank1 = matrix_rank(z_batch, atol=1e-1, rtol=1e-1)
        metrics["rank-3"] = rank3.item()
        metrics["rank-2"] = rank2.item()
        metrics["rank-1"] = rank1.item()

    def single_update(self, memory, batch_size: int):
        """
        H: burn-in len
        L: forward len
        N: TD(n)
        """
        metrics = {}
        losses = 0.0

        # 1. Sample a batch of data in numpy arrays
        (
            batch_burn_in_hist,  # (B, H, O+A+1)
            batch_learn_hist,  # (B, L+N, O+A+1)
            batch_rewards,  # (B, L)
            batch_learn_len,  # (B,) in [1, L]
            batch_forward_idx,  # (B, L) in [N, L+N-1] and {0, <N}
            batch_final_flag,  # (B, L) in {0,1}
            batch_current_act,  # (B, L)
            batch_hidden,  # (B, Z), (B, Z)
            batch_burn_in_len,  # (B,) in [0, H]
            batch_learn_forward_len,  # (B,) in [1, L+N]
            batch_next_obs,  # (B, L, O)
            batch_model_target_reward,  # (B, L)
            batch_model_final_flag,  # (B, L) in {0,1}
            batch_gammas,  # (B, L) in [0, 1]
        ) = memory.sample(batch_size)

        ## 2. Burn-in to get the initial hidden states for learning
        batch_hidden = (
            torch.from_numpy(batch_hidden[0]).view(1, batch_size, -1).to(self.device),
            torch.from_numpy(batch_hidden[1]).view(1, batch_size, -1).to(self.device),
        )
        batch_burn_in_hist = torch.from_numpy(batch_burn_in_hist).to(self.device)
        zero_idx = np.where(batch_burn_in_len == 0)[0]
        batch_burn_in_len[zero_idx] = 1  # temporarily change 0 to 1

        with torch.no_grad():
            _, hidden_burn_in = self.encoder(
                batch_burn_in_hist,
                batch_size,
                batch_hidden,
                self.device,
                list(batch_burn_in_len),
            )  # (1, B, Z), (1, B, Z)
            hidden_burn_in[0][:, zero_idx, :] = batch_hidden[0][:, zero_idx, :]
            hidden_burn_in[1][:, zero_idx, :] = batch_hidden[1][:, zero_idx, :]

            _, target_hidden_burn_in = self.encoder_target(
                batch_burn_in_hist,
                batch_size,
                batch_hidden,
                self.device,
                list(batch_burn_in_len),
            )
            target_hidden_burn_in[0][:, zero_idx, :] = batch_hidden[0][:, zero_idx, :]
            target_hidden_burn_in[1][:, zero_idx, :] = batch_hidden[1][:, zero_idx, :]

        # 3. Forward RNN for a length
        batch_learn_hist = torch.from_numpy(batch_learn_hist).to(self.device)
        ais_z, _ = self.encoder(
            batch_learn_hist,
            batch_size,
            hidden_burn_in,
            self.device,
            list(batch_learn_forward_len),
        )
        unpacked_ais_z, _ = pad_packed_sequence(ais_z, batch_first=True)  # (B,L+N,Z)
        with torch.no_grad():
            target_ais_z, _ = self.encoder_target(
                batch_learn_hist,
                batch_size,
                target_hidden_burn_in,
                self.device,
                list(batch_learn_forward_len),
            )
            target_unpacked_ais_z, _ = pad_packed_sequence(
                target_ais_z, batch_first=True
            )  # (B, L+N, Z)

        q_z = pack_padded_sequence(
            unpacked_ais_z,
            list(batch_learn_len),  # we only use first L hidden states
            batch_first=True,
            enforce_sorted=False,
        )
        if self.aux in ["AIS", "AIS-P2"]:  # modular
            qf = self.critic(q_z.data.detach())  # (*, A)
        else:
            qf = self.critic(q_z.data)  # (*, A)

        self.report_rank(q_z.data.detach(), metrics)

        batch_current_act = torch.from_numpy(batch_current_act).to(self.device)
        packed_current_act = pack_padded_sequence(
            batch_current_act,
            list(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )
        # get Q(h[t], a[t])
        qf = qf.gather(1, packed_current_act.data.view(-1, 1).long())  # (*, 1)

        # 4. (optional) compute auxiliary loss
        batch_final_flag = torch.from_numpy(batch_final_flag).to(self.device)
        packed_final = pack_padded_sequence(
            batch_final_flag,
            list(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )

        if self.aux in ["AIS", "OP"]:  # prepare next_o targets
            next_obs = torch.from_numpy(batch_next_obs).to(self.device)
            next_obs_packed = pack_padded_sequence(
                next_obs,
                list(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )  # o'

        if self.aux in ["AIS", "AIS-P2"]:  # prepare reward targets
            next_rew = (
                torch.from_numpy(batch_model_target_reward)
                .to(self.device)
                .unsqueeze(-1)
            )  # (B, L, 1)
            next_rew_packed = pack_padded_sequence(
                next_rew,
                list(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )  # r
            batch_model_final_flag = torch.from_numpy(batch_model_final_flag).to(
                self.device
            )
            packed_model_final = pack_padded_sequence(
                batch_model_final_flag,
                list(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )

        if self.aux in ["AIS-P2", "ZP"]:  # prepare next_z targets
            q_next_z = pack_padded_sequence(
                unpacked_ais_z[:, 1:],  # shift one step
                list(batch_learn_len),  # we only use first L hidden states
                batch_first=True,
                enforce_sorted=False,
            )
            q_next_z_target = pack_padded_sequence(
                target_unpacked_ais_z[:, 1:],  # shift one step
                list(batch_learn_len),  # we only use first L hidden states
                batch_first=True,
                enforce_sorted=False,
            )

        if self.aux == "AIS":
            AIS_loss = self.compute_AIS_loss(
                metrics,
                batch_z=q_z.data,
                batch_act=packed_current_act.data,
                batch_next_obs=next_obs_packed.data,
                batch_rew=next_rew_packed.data,
                batch_final_flag=packed_final.data,
                batch_model_final_flag=packed_model_final.data,
            )
            self.AIS_optim.zero_grad()
            AIS_loss.backward()
            self.AIS_optim.step()

        elif self.aux == "AIS-P2":
            AIS_loss = self.compute_AISP2_loss(
                metrics,
                batch_z=q_z.data,
                batch_act=packed_current_act.data,
                batch_next_z=q_next_z.data,
                batch_next_z_target=q_next_z_target.data,
                batch_rew=next_rew_packed.data,
                batch_final_flag=packed_final.data,
                batch_model_final_flag=packed_model_final.data,
            )
            self.AIS_optim.zero_grad()
            AIS_loss.backward()
            self.AIS_optim.step()

        elif self.aux == "OP":
            losses += self.aux_coef * self.compute_OP_loss(
                metrics,
                batch_z=q_z.data,
                batch_act=packed_current_act.data,
                batch_next_obs=next_obs_packed.data,
                batch_final_flag=packed_final.data,
            )

        elif self.aux == "ZP":
            losses += self.aux_coef * self.compute_ZP_loss(
                metrics,
                batch_z=q_z.data,
                batch_act=packed_current_act.data,
                batch_next_z=q_next_z.data,
                batch_next_z_target=q_next_z_target.data,
                batch_final_flag=packed_final.data,
            )

        # 5. Compute Target for Double Q-learning
        with torch.no_grad():
            batch_forward_idx = torch.from_numpy(batch_forward_idx).to(self.device)

            ais_z_for_target_action = unpacked_ais_z.gather(
                1,
                batch_forward_idx.view(batch_size, -1, 1)
                .expand(-1, -1, self.AIS_state_size)
                .long(),
            )  # z
            packed_ais_z_for_target_action = pack_padded_sequence(
                ais_z_for_target_action,
                list(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )
            max_idx = self.critic(packed_ais_z_for_target_action.data).max(1)[
                1
            ]  # argmax_a Q(z, a)

            if self.aux in ["AIS", "AIS-P2"]:
                qf_target = self.critic_target(
                    packed_ais_z_for_target_action.data
                )  # Q^tar(z)
            else:
                ais_z_target = target_unpacked_ais_z.gather(
                    1,
                    batch_forward_idx.view(batch_size, -1, 1)
                    .expand(-1, -1, self.AIS_state_size)
                    .long(),  # (B, L, Z)
                )  # (B, L, Z) z^tar
                packed_target_ais = pack_padded_sequence(
                    ais_z_target,
                    list(batch_learn_len),
                    batch_first=True,
                    enforce_sorted=False,
                )
                qf_target = self.critic_target(packed_target_ais.data)  # Q^tar(z^tar)

            qf_target = qf_target.gather(1, max_idx.view(-1, 1).long())  # (*, 1)

            batch_rewards = torch.from_numpy(batch_rewards).to(self.device)
            batch_gammas = torch.from_numpy(batch_gammas).to(self.device)

            packed_reward = pack_padded_sequence(
                batch_rewards,
                list(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_gammas = pack_padded_sequence(
                batch_gammas,
                list(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )

            next_q_value = (
                packed_reward.data
                + packed_gammas.data * packed_final.data * qf_target.view(-1)
            )

        # 6. Sum Q-learning loss and Auxiliary loss
        if self.args["TD_loss"] == "mse":
            qf_loss = F.mse_loss(qf.view(-1), next_q_value, reduce=False)
        elif self.args["TD_loss"] == "smooth_l1":
            qf_loss = F.smooth_l1_loss(qf.view(-1), next_q_value, reduce=False)

        qf_loss = qf_loss.mean()
        metrics["critic_loss"] = qf_loss.item()
        losses += qf_loss

        self.optim.zero_grad()
        if self.aux in ["ZP", "OP"]:
            self.AIS_optim.zero_grad()

        losses.backward()

        self.optim.step()
        if self.aux in ["ZP", "OP"]:
            self.AIS_optim.step()

        return metrics

    def compute_AIS_loss(
        self,
        metrics,
        batch_z,
        batch_act,
        batch_next_obs,
        batch_rew,
        batch_final_flag,
        batch_model_final_flag,
    ):
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)

        predicted_obs, predicted_rew = self.model(model_input)
        obs_squared_error = (predicted_obs - batch_next_obs) ** 2
        rew_squared_error = (predicted_rew - batch_rew) ** 2
        obs_loss = (
            obs_squared_error.sum(-1) * batch_final_flag
        ).mean() / self.obs_dim  # normalized
        rew_loss = (rew_squared_error.sum(-1) * batch_model_final_flag).mean()

        metrics["op_loss"] = obs_loss.item()
        metrics["rp_loss"] = rew_loss.item()

        return self.aux_coef * obs_loss + rew_loss

    def compute_AISP2_loss(
        self,
        metrics,
        batch_z,
        batch_act,
        batch_next_z,
        batch_next_z_target,
        batch_rew,
        batch_final_flag,
        batch_model_final_flag,
    ):
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)

        if self.aux_optim == "online":
            true_next_z = batch_next_z
        elif self.aux_optim == "detach":
            true_next_z = batch_next_z.detach()
        elif self.aux_optim == "ema":
            true_next_z = batch_next_z_target
        else:
            raise ValueError(self.aux_optim)

        predicted_next_z, predicted_rew = self.model(model_input)  # z', r

        z_squared_error = (predicted_next_z - true_next_z) ** 2
        rew_squared_error = (predicted_rew - batch_rew) ** 2
        z_loss = (
            z_squared_error.sum(-1) * batch_final_flag
        ).mean() / self.AIS_state_size  # normalized
        rew_loss = (rew_squared_error.sum(-1) * batch_model_final_flag).mean()

        metrics["zp_loss"] = z_loss.item()
        metrics["rp_loss"] = rew_loss.item()

        return self.aux_coef * z_loss + rew_loss

    def compute_OP_loss(
        self,
        metrics,
        batch_z,
        batch_act,
        batch_next_obs,
        batch_final_flag,
    ):
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)

        predicted_obs = self.model(model_input)
        squared_error = (predicted_obs - batch_next_obs) ** 2
        model_loss = squared_error.sum(-1) * batch_final_flag

        model_loss = model_loss.mean() / self.obs_dim  # normalized

        metrics["op_loss"] = model_loss.item()

        return model_loss

    def compute_ZP_loss(
        self,
        metrics,
        batch_z,
        batch_act,
        batch_next_z,
        batch_next_z_target,
        batch_final_flag,
    ):
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)
        predicted_next_z = self.model(model_input)  # z'

        if self.aux_optim == "online":
            true_next_z = batch_next_z
        elif self.aux_optim == "detach":
            true_next_z = batch_next_z.detach()
        elif self.aux_optim == "ema":
            true_next_z = batch_next_z_target
        else:
            raise ValueError(self.aux_optim)

        squared_error = (predicted_next_z - true_next_z) ** 2
        model_loss = squared_error.sum(-1) * batch_final_flag

        model_loss = model_loss.mean() / self.AIS_state_size  # normalized

        metrics["zp_loss"] = model_loss.item()

        return model_loss
