import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from typing import NamedTuple, Optional

import pickle

from absl import app
from absl import flags

import numpy as np
import pandas as pd

import jax

import optax

from replearn import learn
from replearn import loadunload
from replearn import features
from replearn import rollout


if __name__ == "__main__":
    flags.DEFINE_string("results_dir", "./results", "Directory used to log results.")

FLAGS = flags.FLAGS

# for i in {1..100}; do echo $(od -A n -t u -N 4 /dev/urandom | tr -d ' \n'); done
SEEDS = [
    1843764854,
    2953627709,
    2065696022,
    3917761262,
    1592259201,
    684225940,
    3906814804,
    855070892,
    79122374,
    3353291887,
    2001425368,
    3649968832,
    1137905990,
    1274987999,
    1821861786,
    3683081310,
    1087561443,
    310321600,
    1055175732,
    121547637,
    4044866360,
    182248956,
    4039913460,
    462825347,
    3727679027,
    3215526904,
    2431647752,
    2379353061,
    2323226982,
    3725743208,
    2031918674,
    3762025650,
    425696606,
    805171965,
    2503275839,
    2277247045,
    2109158367,
    181242376,
    3956246306,
    2755595351,
    4187306323,
    1007867152,
    1242463926,
    4129796788,
    1099410125,
    209730990,
    64549074,
    712869140,
    3522339780,
    3428373530,
    2464126123,
    3456720685,
    503202288,
    518482939,
    862737849,
    2403136178,
    159923561,
    2839661397,
    2140359683,
    2108678269,
    1984270380,
    678399733,
    358224968,
    4124224329,
    3459659839,
    3008777333,
    1884818714,
    2158764360,
    3267115782,
    1498615144,
    729227282,
    356343867,
    3273136234,
    600066107,
    3613546418,
    1637623759,
    1043304407,
    2854775057,
    2055801193,
    1136497228,
    1506477464,
    3358102518,
    3061257360,
    3644648965,
    3559804656,
    3972212350,
    963994575,
    1947277982,
    4279881374,
    156505623,
    2226220832,
    4149186854,
    1200204930,
    1194710917,
    3409768682,
    2569256998,
    3500793866,
    606886659,
    2720394887,
    2696168484,
]


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print(f"Using result dir: {FLAGS.results_dir}")

    action_encoder = features.OneHot(2)
    observation_encoder = features.OneHot(3)
    history_encoder = features.TruncatedHistoryEncoder(20)

    encoder = learn.create_latent_encoder(2)
    optimizer = optax.sgd(1e-2)

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"Starting run {i} with seed={seed}")
        data_seed, train_seed = np.random.SeedSequence(seed).spawn(2)
        s_t, a_t, s_tp1 = rollout.rollout_dataset(
            data_seed.generate_state(2),
            env_cls=loadunload.LoadUnload,
            policy=loadunload.LoadUnloadPolicy(None),
            history_encoder=history_encoder,
            act_encoder=action_encoder,
            obs_encoder=observation_encoder,
            max_traj_length=200,
            num_traj=10,
        )
        for use_stop_gradient in ["Online", "Detached", "EMA"]:
            logs = learn.train(
                key=train_seed.generate_state(2),
                optimizer=optimizer,
                encoder=encoder,
                states=s_t,
                actions=a_t,
                next_states=s_tp1,
                num_steps=500,
                log_n_steps=10,
                use_stop_gradient=use_stop_gradient,
            )
            for log in logs:
                log.update({"seed": seed, "use_stop_gradient": use_stop_gradient})
            results = results + logs

    with open(os.path.join(FLAGS.results_dir, "loadunload.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    app.run(main)
