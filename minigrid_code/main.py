import os
import time

pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

import argparse
import json
from run import run_exp
import logger


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env_name", type=str)

    # Training
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--num_steps", type=int, default=4000000)
    ## Freq
    parser.add_argument("--logging_freq", type=int, default=10000)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--rl_update_every_n_steps", type=int, default=10)
    parser.add_argument("--rl_updates_per_step", type=int, default=1)
    parser.add_argument("--model_updates_per_step", type=int, default=1)
    parser.add_argument("--random_actions_until", type=int, default=0)

    # Buffer
    parser.add_argument("--replay_size", type=int, default=400000)
    parser.add_argument("--batch_size", type=int, default=256)
    ## Len
    parser.add_argument("--burn_in_len", type=int, default=50)
    parser.add_argument("--learning_obs_len", type=int, default=10)
    parser.add_argument("--forward_len", type=int, default=5)

    # Representation learning
    parser.add_argument("--aux", type=str, default="None")
    parser.add_argument("--aux_optim", type=str, default="None")
    parser.add_argument("--aux_coef", type=float, default=0.5)
    parser.add_argument("--aux_lr", type=float, default=1e-3)
    parser.add_argument("--AIS_state_size", type=int, default=128)

    # RL (DDQN)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--rl_lr", type=float, default=1e-3)
    parser.add_argument("--TD_loss", type=str, default="mse")
    ## Exploration
    parser.add_argument("--EPS_start", type=float, default=1.0)
    parser.add_argument("--EPS_end", type=float, default=0.05)
    parser.add_argument("--EPS_decay", type=int, default=400000)
    parser.add_argument("--EPS_decay_type", type=str, default="exponential")
    parser.add_argument("--test_epsilon", type=float, default=0.0)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    print(params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    format_strs = ["csv"]
    save_dir = "logs"
    if args.debug:
        save_dir = "debug"
        format_strs.extend(["stdout", "log"])  # logger.log

    unique_id = time.strftime("%Y-%m-%d-%H:%M:%S") + "_" + jobid + "-" + pid
    logdir = os.path.join(save_dir, args.env_name, unique_id)
    params["logdir"] = logdir
    logger.configure(dir=logdir, format_strs=format_strs)

    config_path = os.path.join(logdir, "config.json")
    with open(config_path, "w") as fp:
        json.dump(params, fp, indent=4)

    ###################
    ### RUN TRAINING
    ###################

    if params["seed"] < 0:
        params["seed"] = int(
            pid
        )  # to avoid conflict within a job which has same datetime
    run_exp(params)


if __name__ == "__main__":
    main()
