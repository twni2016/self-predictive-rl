# Code for State Representation Learning in Standard and Distracting MDPs (Section 5.1 & 5.2)

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

Our minimalist $\phi_L$: set `algo=ours` and
- `aux`: select from `fkl, rkl, l2`
- `aux_optim`: select from `ema, detach, online`

Learning $\phi_O$ and $\phi_{Q^*}$: set `algo=ours` and
- `aux`: select from `op-l2, op-kl, null`

## Examples

To reproduce original ALM(3) on Humanoid-v2:
```bash
python train.py id=Humanoid-v2 algo=alm-3
```

To reproduce our minimalist $\phi_L$ with l2 objective and EMA targets on Ant-v2:
```bash
python train.py id=Ant-v2 algo=ours aux=l2 aux_optim=ema aux_coef=v-10.0
```

To reproduce our minimalist $\phi_L$ with reverse KL objective and online targets on Ant-v2:
```bash
python train.py id=Ant-v2 algo=ours aux=rkl aux_optim=online aux_coef=v-1.0
```

To reproduce learning $\phi_O$ with forward KL objective on distracted HalfCheetah-v2 with 256 distractors:
```bash
python train.py id=HalfCheetah-v2 distraction=256 scale=1.0 algo=ours aux=op-kl aux_optim=null aux_coef=v-1.0
```

To reproduce learning $\phi_{Q^*}$ on distracted HalfCheetah-v2 with 256 distractors:
```bash
python train.py id=HalfCheetah-v2 distraction=256 scale=1.0 algo=ours aux=null aux_optim=null aux_coef=v-0.0
```

You will see the logging and executed config files in `logs/` folder.

## Logged Results and Plotting

The log files used in our paper is provided at [Google Drive](https://drive.google.com/file/d/1KaxHySEX3xNCfqUyMsPM2sLzo96SQZd5/view?usp=sharing) (~2.8GB; maybe redundant with some unpublished results). You can download and unzip it to this folder and name it as `logs`.

We use the [`vis.ipynb`](https://github.com/twni2016/self-predictive-rl/blob/main/mujoco_code/vis.ipynb) for generating plots in our paper. 
Below are the commands to generate specific figures in the paper. 

### Plotting Standard MuJoCo results
In Part 1, in product(), choose `[0, ]` in `distraction`

- Figure 3: learning curves. Choose 
```python
metric, y_label, sci_axis = "return", "episode return", "both"
tag = ""
hue = "aux"
style = None
# in query_fn(), select the line "return flags["algo"] == "alm-3""
```
- Figure 4, 11, and 12: ablation on ZP targets. Choose
```python
metric, y_label, sci_axis = "return", "episode return", "both"
# metric, y_label, sci_axis = "rank-2", "matrix rank", "x"
# metric, y_label, sci_axis = "l2", "ZP loss", "x"
tag = "l2" # "fkl", "rkl"
hue = "aux_optim"
style = None
# in query_fn(), select the line "return False"
```
- Figure 10: ablation on ALM variants. Choose
```python
metric, y_label, sci_axis = "return", "episode return", "both"
tag = "ablate-"
hue = "aux"
style = None
# in query_fn(), select the line "return flags["algo"] in ["alm-3", "alm-no-model", "alm-no-model-1"]"
```

### Plotting Distracting MuJoCo results

First, in query_fn(), select the line `return False`; in product(), choose `[2**4, 2**5, 2**6, 2**7, 2**8]` in `distraction`. 

- Figure 13: learning curves. In Part 1,
```python
plt.rcParams["axes.titlesize"] = 11  # for distractors
metric, y_label, sci_axis = "return", "episode return", "both"
tag = ""
hue = "aux"
style = None
```
- Figure 5, aggregated plots. In Part 2,
```python
metric, y_label, sci_axis = "return", "episode return", "y"
hue = "aux"
```

## Acknowledgement
Our codebase has been largely build on Raj's codebase [ALM](https://github.com/RajGhugare19/alm).
