# Bridging State and History Representations: Understanding Self-Predictive RL
This is the official code for the paper

["Bridging State and History Representations: Understanding Self-Predictive RL"](https://arxiv.org/abs/2401.08898), **ICLR 2024**

by [Tianwei Ni](https://twni2016.github.io/), [Benjamin Eysenbach](https://ben-eysenbach.github.io/), [Erfan Seyedsalehi](https://openreview.net/profile?id=~Erfan_Seyedsalehi2), [Michel Ma](https://scholar.google.com/citations?user=capMFX8AAAAJ&hl=en), [Clement Gehring](https://people.csail.mit.edu/gehring/), [Aditya Mahajan](https://cim.mcgill.ca/~adityam/), and [Pierre-Luc Bacon](http://pierrelucbacon.com/). 

## TLDR: A Minimal Augmentation for Model-Free RL Loss 🚀 

In this work, we demonstrate a *principled, minimal, and effective* design, as reflected in the following pseudocode:

```python
def total_loss(hist, act, next_obs, rew):
    """
    Compute the total loss for learning one of the three abstractions.

    Args: Batch of transition data (h, a, o', r).
        hist h: (B, T, O+A), act a: (B, A), next_obs o': (B, O), rew r: (B, 1)
    """

    # Encode current history into a latent state
    h_enc = Encoder(hist)  # z: (B, Z)
    next_hist = torch.cat([hist, torch.cat([act, next_obs], dim=-1)], dim=1)  # h': (B, T+1, O+A)
    # Encode next history into a latent state using an EMA encoder
    next_h_enc_tar = Encoder_Target(next_hist)  # z': (B, Z)

    # Model-free RL loss in the latent state space (e.g., TD3, R2D2)
    rl_loss = RL_loss(h_enc, act, next_h_enc_tar, rew)  # (z, a, z', r)

    if [learning Q^*-irrelevance representations]:  # model-free RL
      return rl_loss
    elif [learning self-predictive representations]:  # l2 loss with EMA ZP target
      zp_loss = ((Latent_Model(h_enc, act) - next_h_enc_tar)**2).sum(-1).mean()  
      return rl_loss + coef * zp_loss
    elif [learning observation-predictive representations]:  # l2 loss
      op_loss = ((Observ_Model(h_enc, act) - next_obs)**2).sum(-1).mean()
      return rl_loss + coef * op_loss
``` 

## Background 🔍 

In deep RL, numerous representation learning methods have been proposed, ranging from *state representations* for MDPs to *history representations* for POMDPs. However, these methods often involve different learning objectives and training techniques, making it challenging for RL practitioners to select the most suitable approach for their specific problems.

This work unifies various representation learning methods by analyzing their objectives and ideal abstractions. Surprisingly, these methods are connected by a **self-predictive** condition, termed the **ZP condition**: *the latent state generated by the encoder can be used to predict the next latent state*. We summarize three abstractions learned by these methods and provide examples of popular instances:

1. **$Q^*$-irrelevance abstraction**: purely maximizes returns. Examples: [model-free RL (cleanrl)](https://github.com/vwxyzjn/cleanrl), [recurrent model-free RL](https://github.com/twni2016/pomdp-baselines).
2. **Self-predictive abstraction**: involves the self-predictive (ZP) and reward-prediction (RP) conditions. Examples: [SPR](https://github.com/mila-iqia/spr), [DBC](https://github.com/facebookresearch/deep_bisim4control), [TD-MPC](https://github.com/nicklashansen/tdmpc), [EfficientZero](https://github.com/YeWR/EfficientZero). 
3. **Observation-predictive abstraction**: involves the observation-predictive (OP) and reward-prediction (RP) conditions. Examples: [Dreamer](https://github.com/danijar/dreamerv3), [SLAC](https://github.com/alexlee-gk/slac), [SAC-AE](https://github.com/denisyarats/pytorch_sac_ae).

## Using Our Minimalist Algorithm as Your Baseline 🔧 

In our paper, we establish how the ZP condition connects the three abstractions. Crucially, we investigate the training objectives for learning ZP, including widely-used $\ell_2$, $\cos$, and KL divergences, along with the *stop-gradient* operator to prevent representational collapse.

These analyses lead to the development of **our minimalist algorithm** for learning self-predictive abstraction. We provide the code as **a baseline** for future research, believing it to be:
- **Principled in representation learning**: targets each of the three abstractions.
- **Minimal in algorithmic design**: uses single auxiliary task for representation learning (just one extra loss), and model-free policy optimization (no planning).
- **Effective in practice**: our implementation of self-predictive representations outperforms $Q^*$-irrelevance abstraction (the model-free baseline), and is more robust to distractions than observation-predictive representations.

## Code Implementation 🗂️

- [`mujoco_code/`](https://github.com/twni2016/self-predictive-rl/tree/main/mujoco_code): contains the code on standard MDPs (Section 5.1) and distracting MDPs (Section 5.2) using [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) simulators.
- [`minigrid_code/`](https://github.com/twni2016/self-predictive-rl/tree/main/minigrid_code): contains the code on sparse-reward POMDPs (Section 5.3) using [MiniGrid](https://minigrid.farama.org/index.html) environments.
- [`linear_code/`](https://github.com/twni2016/self-predictive-rl/tree/main/linear_code): contains the code for illustrating our theorem on stop-gradient to prevent collapse (Section 4.2).

## Our Recommendations for Practitioners 📋 

Here we restate our preliminary recommendations from our paper (Section 6):

- Analyze your task first. For example, in noisy or distracting tasks, consider using self-predictive representations. In sparse-reward tasks, consider using observation-predictive representations. In deterministic tasks, choose the deterministic $\ell_2$ objectives for representation learning.  
- Use our minimalist algorithm as your baseline. Our algorithm allows for an independent evaluation of representation learning and policy optimization effects.  Start with end-to-end learning and model-free RL for policy optimization. 
- Implementation tips. For our minimalist algorithm, we recommend adopting the $\ell_2$ objective with EMA ZP targets first. When tackling POMDPs, start with recurrent networks as the encoder.

## Questions❓

If you have any questions, please raise an issue (preferred) or send an email to Tianwei (tianwei.ni@mila.quebec).

## Citation

```bibtex
@inproceedings{ni2024bridging,
  title={Bridging State and History Representations: Understanding Self-Predictive RL},
  author={Ni, Tianwei and Eysenbach, Benjamin and Seyedsalehi, Erfan and Ma, Michel and Gehring, Clement and Mahajan, Aditya and Bacon, Pierre-Luc},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
