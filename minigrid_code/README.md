# Code for History Representation Learning in Sparse-Reward POMDPs (Section 5.3)

Code contributors: [Erfan Seyedsalehi](https://openreview.net/profile?id=~Erfan_Seyedsalehi2) (main), [Tianwei Ni](https://twni2016.github.io/).

Benchmark: [MiniGrid](https://minigrid.farama.org/environments/minigrid/
) benchmark composed of 20 tasks, featuring sparse rewards and partial observability.

Baseline: a single-thread version of [R2D2](https://openreview.net/forum?id=r1lyTjAqYX), named as R2D2 below.


## Installation

We use python 3.7+ and list the basic requirements in `requirements.txt`.


## Examples

To reproduce R2D2 in SimpleCrossingS9N1: 
```bash
python main.py --num_steps 4000000 --env_name MiniGrid-SimpleCrossingS9N1-v0  \
--aux None
```

To reproduce our minimalist algorithm (end-to-end ZP with EMA target) in SimpleCrossingS9N1: 
```bash
python main.py --num_steps 4000000 --env_name MiniGrid-SimpleCrossingS9N1-v0  \
--aux ZP --aux_coef 1.0 --aux_optim ema
```

To reproduce the minimalist algorithm (end-to-end OP) in SimpleCrossingS9N1: 
```bash
python main.py --num_steps 4000000 --env_name MiniGrid-SimpleCrossingS9N1-v0  \
--aux OP --aux_coef 0.01
```

To reproduce the phased algorithm (RP + ZP with EMA) in SimpleCrossingS9N1: 
```bash
python main.py --num_steps 4000000 --env_name MiniGrid-SimpleCrossingS9N1-v0  \
--aux AIS-P2 --aux_coef 1.0
```

To reproduce the phased algorithm (RP + OP) in SimpleCrossingS9N1: 
```bash
python main.py --num_steps 4000000 --env_name MiniGrid-SimpleCrossingS9N1-v0  \
--aux AIS --aux_coef 1.0
```

## Plotting and Logged Results
TODO

## Flags

This program accepts the following command line arguments:

| Option          | Description |
| --------------- | ----------- |
| `--aux` |  This specifies whether model-learning is done or not. 'AIS' is for model learning (RQL-AIS). 'None' is for no model learning (ND-R2D2). |
| `--env_name` | The environment name.  |
| `--batch_size` | The batch size used for AIS updates and reinforcement learning updates. This specifies the number of samples drawn from the buffer. Each trajectory has a fixed length (learning_obs_len) |
| `--hidden_size` | The number of neurons in the hidden layers of the Q network. |
| `--gamma` | Discount Factor |
| `--AIS_state_size` | The size of the hidden vector and the output of the LSTM used as state representation for the POMDP. |
| `--rl_lr` | The learning rate used for updating Q networks and the LSTMs (for ND-R2D2) and only the Q-network (for RQL-AIS) |
| `--aux_lr` | The learning rate used for updating the AIS components (for RQL-AIS). |
| `--num_steps` | Total number of training steps taken in the environment.|
| `--target_update_interval` |  This specifies the environment step intervals after which the target Q network (and target LSTM in case of ND-R2D2) is updated. |
| `--replay_size` |  This spcecifies the number of episodes that are stored in the replay memory. After the replay buffer is filled, new experience episodes will overwrite the least recenet episodes in the buffer. |
| `--aux_coef` |  The hyperparameter which specifies how we are averaging between reward learning loss and next observation predictions loss in the AIS learning phase. |
| `--logging_freq` |  The frequency in terms of environment steps in which we evaluate the agent, log the results and save the neural network parameters on disk. |
| `--rl_update_every_n_steps` |  It specifies the frequency in terms of environment steps at which we do reinforcement learning updates. |
| `--EPS_start` |  This specifies the start value for the epsilon hyperparameter used in Q-learning for exploration. |
| `--EPS_decay` |  This specifies decay rate for the epsilon hyperparameter used in Q-learning for exploration |
| `--EPS_end` | This specifies the end value for the epsilon hyperparameter used in Q-learning for exploration |
| `--burn_in_len` | Length of the preceding Burn-In Sequence saved with each sample in the R2D2 buffer. |
| `--learning_obs_len` | Sequence length of R2D2 samples. |
| `--forward_len` | The multi-step Q-learning length. |
| `--test_epsilon` | Epsilon value used at test time. Default is 0. |

## Acknowledgement

Our codebase has been largely build on Erfan's codebase [RQL-AIS](https://github.com/esalehi1996/POMDP_RL).
