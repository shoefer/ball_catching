# ball-catching

#### python framework for studying control strategies to Chapman's ball catching problem

This material accompanies my paper *No Free Lunch in Ball Catching: A Comparison of Cartesian and Angular
Representations for Control* as well as my doctoral thesis *On Decomposability in Robot Reinforcement Learning*.
&copy; 2017 Sebastian Höfer

Includes jupyter notebooks providing proofs included in the thesis/paper that were verified with sympy.

## Proofs for Section 4.2: Heuristics for Ball Catching

Proofs can be found in ```ipynb/proofs-chapman.ipynb```

## Proofs for Appendix B.1: Angular Representation Violates Markov Property

Proofs can be found in ```ipynb/proof-angular-non-markov.ipynb```


## Quickstart

We recommend to use [anaconda](https://www.continuum.io/downloads) which is an easy-to-install complete python bundle
that contains all packages required.

1) Check out ball-catching from the repository

    git clone https://github.com/shoefer/ball_catching.git

2) Create conda environment

    conda create -n bc python=2.7
    source activate bc

3) Install

    pip install -e .

4) Run experiments!

### Single strategy under single condition

Angular controller (COVOAC) in 2D without noise:

    python run.py single --strategies COVOAC2DStrategy

Cartesian controller (LQG) in 2D without noise:

    python run.py single --strategies LQGStrategy

### Multiple strategies with multiple initial conditions

This command will run a comparison of one or multiple strategies under different settings:

    python run.py multi --range s --strategies LQGStrategy COVOAC2DStrategy

It will run for a while and then automatically generate experimental data and show plots (unless you pass
```--no_plot```). The experimental data will be stored in ```~/ball_catching_data``` (you can change this in config.py).
For every experiment you run you will find a new folder with this structure (with different timestamps):

    2DBallCatching__2018-04-22_22-04-54-399560
    - cmd.txt
    - COVOAC2DStrategy_2D_ideal_DTinv-60_2018-04-22_22-04-54-400944
      - BC_COVOAC2DStrategy_2018-04-22_22-04-54-402504
      - BC_COVOAC2DStrategy_2018-04-22_22-04-54-836147
      - ...
    - LQGStrategy_2D_ideal_DTinv-60_2018-04-22_22-04-54-400352
      - BC_LQGStrategy_2018-04-22_22-05-01-573956
      - BC_LQGStrategy_2018-04-22_22-05-04-440498
      - ...

The root folder is called *experiment set folder* (2DBallCatching...),
the second level is called *experiment folder* (COVOAC2DStrategy..., LQGStrategy_2D...) and the last level is called
*trial folder*. There are different plotting scripts that let you look at all three levels:

Plot statistics for all strategies across all initial conditions:

    python plotting/multi_experiment_set_plot.py 2DBallCatching__2018-04-22_22-04-54-399560

Compare overall performance of strategies:

    python plotting/comparison_plot.py 2DBallCatching__2018-04-22_22-04-54-399560 --metrics distance

All plots will also be stored as PDF inside the experiment folders.

Plot statistics only for a particular strategy:

    python plotting/multi_experiment_set_plot.py 2DBallCatching__2018-04-22_22-04-54-399560/COVOAC2DStrategy_2D_ideal_DTinv-60_2018-04-22_22-04-54-400944

Inspect the performance of the strategy in a particular initial condition:

    python plotting/single_experiment_plot.py 2DBallCatching__2018-04-22_22-04-54-399560/COVOAC2DStrategy_2D_ideal_DTinv-60_2018-04-22_22-04-54-400944/BC_COVOAC2DStrategy_2018-04-22_22-04-54-402504


#### Other experiments

You have full control over all types of experiments you want to run. To get help on all possible variants run

    python run.py --help


### Running MPCStrategy

https://github.com/b4be1/easy_catch

1) Install casadi 2.4.3

- Download from https://sourceforge.net/projects/casadi/files/CasADi/2.4.3/
- Append location to "casadi-py27-np1.9.1-v2.4.3" to your python path
-




#### Requirements

- [SymPy](http://www.sympy.org/en/index.html)
- NumPy
- Matplotlib
- Jupyter (Notebook)

Follow the installation requirements for these frameworks.


