# Code for state representation learning in standard and distracting MDPs (Section 5.1 & 5.2)

Code contributor: [Tianwei Ni](https://twni2016.github.io/).

## Installation

We use python 3.7+ and list the basic requirements in `requirements.txt`.

## Key Flags
The configuration file is `cfgs/agent/alm.yaml`.

Environments:
- **Standard MuJoCo** (Section 5.1): set `id` from "HalfCheetah-v2", "Humanoid-v2", "Ant-v2", "Walker2d-v2", "Hopper-v2"
- **Distracted MuJoCo** (Section 5.2): set `distraction=128` and `scale=1.0` for 128 distractors with standard Gaussian noises

Compared algorithms:
- ALM(3): set `algo=alm-3`
- ALM-no-model: set `algo=alm-no-model`
- ALM(0): set `algo=alm-0`

Our minimalist phi_L: set `algo=ours` and
- `aux`: select from `fkl, rkl, l2`
- `aux_optim`: select from `ema, detach, online`

Learning phi_O and phi_Q*: set `algo=ours` and
- `aux`: select from `op-l2, op-kl, null`

## Examples

To reproduce original ALM(3) on Humanoid-v2:
```bash
python train.py id=Humanoid-v2 algo=alm-3
```

To reproduce our minimalist phi_L with l2 objective and EMA targets on Ant-v2:
```bash
python train.py id=Ant-v2 algo=ours aux=l2 aux_optim=ema aux_coef=v-10.0
```

To reproduce our minimalist phi_L with reverse KL objective and online targets on Ant-v2:
```bash
python train.py id=Ant-v2 algo=ours aux=rkl aux_optim=online aux_coef=v-1.0
```

To reproduce learning phi_O with forward KL objective on distracted HalfCheetah-v2 with 256 distractors:
```bash
python train.py id=HalfCheetah-v2 distraction=256 scale=1.0 algo=ours aux=op-kl aux_optim=null aux_coef=v-1.0
```

To reproduce learning phi_Q* on distracted HalfCheetah-v2 with 256 distractors:
```bash
python train.py id=HalfCheetah-v2 distraction=256 scale=1.0 algo=ours aux=null aux_optim=null aux_coef=v-0.0
```

You will see the logging and executed config files in `logs/` folder.

## Plotting and Logged Results
TODO

## Acknowledgement
Our codebase has been largely build on Raj's codebase [ALM](https://github.com/RajGhugare19/alm).
