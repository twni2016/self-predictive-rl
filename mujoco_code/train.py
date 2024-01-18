import os
import time

pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

from utils import logger, system
from omegaconf import DictConfig, OmegaConf
import hydra
import warnings

warnings.simplefilter("ignore", UserWarning)


@hydra.main(config_path="cfgs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.benchmark == "gym":
        from workspaces.mujoco_workspace import MujocoWorkspace as W
    else:
        raise NotImplementedError
    env_id = cfg.id
    if cfg.distraction > 0:
        env_id += f"-d{cfg.distraction}"

    if cfg.seed < 0:
        cfg.seed = int(pid)  # to avoid conflict within a job which has same datetime

    run_name = f"{system.now_str()}+{jobid}-{pid}"
    format_strs = ["csv"]
    if cfg.debug:
        cfg.save_dir = "debug"
        format_strs.extend(["stdout", "log"])  # logger.log

    log_path = os.path.join(cfg.save_dir, env_id, run_name)
    logger.configure(dir=log_path, format_strs=format_strs, precision=4)

    existing_variants = {
        "alm-3": (False, False, False, False, "rkl", "ema", "v-1.0", 3),
        "alm-1": (False, False, False, False, "rkl", "ema", "v-1.0", 1),
        "alm-no-model": (False, True, False, False, "rkl", "ema", "v-1.0", 1),
        "alm-0": (True, True, False, False, "rkl", "ema", "v-1.0", 1),
        "alm-0-ours": (True, True, True, True, "rkl", "ema", "v-1.0", 1),
        "td3": (True, True, True, True, None, None, "v-0.0", 1),
    }

    if cfg.algo in existing_variants:
        (
            cfg.disable_reward,
            cfg.disable_svg,
            cfg.freeze_critic,
            cfg.online_encoder_actorcritic,
            cfg.aux,
            cfg.aux_optim,
            cfg.aux_coef,
            cfg.seq_len,
        ) = existing_variants[cfg.algo]
    elif cfg.algo == "ours":
        (
            cfg.disable_reward,
            cfg.disable_svg,
            cfg.freeze_critic,
            cfg.online_encoder_actorcritic,
            cfg.seq_len,
        ) = (True, True, True, True, 1)
    else:
        raise ValueError(cfg.algo)

    # write config to a yml
    with open(os.path.join(log_path, "flags.yml"), "w") as f:
        OmegaConf.save(cfg, f)

    workspace = W(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
