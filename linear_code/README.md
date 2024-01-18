# Code for Illustrating Theorem 3

Code contributors: [Clement Gehring](https://people.csail.mit.edu/gehring/) (main), [Tianwei Ni](https://twni2016.github.io/).

## Installation 
```bash
python setup.py
```

## Running 

```bash
python experiments/run_mountaincar.py
python experiments/run_loadunload.py
```
For each environment, it will run each option of ZP target ("Online", "Detached", "EMA") for 100 seeds and generate the resulting data in `results/` folder.

## Plotting

```bash
python experiments/plot.py \
    --results_path results/mountaincar.pkl --figure_title "Mountain Car"
python experiments/plot.py \
    --results_path results/loadunload.pkl --with_legend --figure_title "Load-Unload"
```

It will generate the Figure 2 in the paper.


