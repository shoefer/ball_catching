# ball_catching

#### python framework for studying control strategies to Chapman's ball catching problem

This package accompanies the paper

[Höfer, Raisch, Toussaint, Brock.
No Free Lunch in Ball Catching: A Comparison of Cartesian and Angular Representations for Control.
2018]()

as well as my doctoral thesis

[Sebastian Höfer. On decomposability in robot reinforcement learning. Dissertation. Technische Universität Berlin, Germany, June 2017.](http://dx.doi.org/10.14279/depositonce-6054)

It contains

- A lightweight python library for running ball catching experiments, including implementations of all
  control strategies presented in the paper.
- jupyter notebooks with proofs included in the paper (verified proofs, using sympy).

## Quickstart

We recommend to use [anaconda](https://www.continuum.io/downloads) which is an easy-to-install complete python bundle
that contains all packages required.

1) Check out ball-catching from the repository

    git clone https://github.com/shoefer/ball_catching.git ball_catching

2) Create conda environment

    conda create -n bc python=2.7
    source activate bc

3) Install

    cd ball_catching        # need to be in root folder of git repo
    pip install -e .

4) Run experiments!

### Single strategy under single condition

*Angular controller (COV-OAC)* in 2D without noise:

    python ball_catching/run.py single --strategies COVOAC2DStrategy

It should pop up a couple plots and return an console output similar to this:

    Statistics: (trials 1)
    mean(terminal_distance) -> 0.004031813928037309
    std(terminal_distance) -> 0.0
    mean(terminal_velocity) -> 3.9384176863998617
    std(terminal_velocity) -> 0.0
    mean(control_effort) -> 40.749233121618374
    std(control_effort) -> 0.0
    mean(duration) -> 4.3247960696889205
    std(duration) -> 0.0
    Agent velocity at 4.324796 s 3.938418

*Cartesian controller (LQG)* in 2D without noise:

    python ball_catching/run.py single --strategies LQGStrategy

### Multiple strategies with multiple initial conditions

This command will run a comparison of one or multiple strategies under different settings:

    python ball_catching/run.py multi --range m --strategies LQGStrategy COVOAC2DStrategy

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

    python ball_catching/plotting/multi_experiment_set_plot.py 2DBallCatching__2018-04-22_22-04-54-399560

Compare overall performance of strategies:

    python ball_catching/plotting/comparison_plot.py 2DBallCatching__2018-04-22_22-04-54-399560 --metrics distance

All plots will also be stored as PDF inside the experiment folders.

Plot statistics only for a particular strategy:

    python ball_catching/plotting/multi_experiment_set_plot.py 2DBallCatching__2018-04-22_22-04-54-399560/COVOAC2DStrategy_2D_ideal_DTinv-60_2018-04-22_22-04-54-400944

Inspect the performance of the strategy in a particular initial condition:

    python ball_catching/plotting/single_experiment_plot.py 2DBallCatching__2018-04-22_22-04-54-399560/COVOAC2DStrategy_2D_ideal_DTinv-60_2018-04-22_22-04-54-400944/BC_COVOAC2DStrategy_2018-04-22_22-04-54-402504

### 3D experiments

For example, angular COV-OAC strategy:

    python ball_catching/run.py single3d --strategies COVOAC3DStrategy

or LQG:

    python ball_catching/run.py single3d --strategies LQGStrategy

### Adding noise

Compare performance of strategies under drag:

    python ball_catching/run.py multi3d --range m --strategies LQGStrategy COVIO3DStrategy --noise drag --noise_only

### Other experiments

You have full control over all types of experiments you want to run. To get help on all possible variants run

    python ball_catching/run.py --help

## Running MPCStrategy

In order to run the model-predictive control strategy in belief space [Belousov, 2016], you need to install casadi,
an autodiff framework. The authors original source code is used: [easy_catch](https://github.com/b4be1/easy_catch)

### Install casadi 2.4.3

- Go to https://sourceforge.net/projects/casadi/files/CasADi/2.4.3/ and download the py27 binary for your OS, e.g.
  for MacOS get ```casadi-py27-np1.9.1-v2.4.3.tar.gz```

- Extract to some location in your workspace, e.g.

      mkdir -p ~/Workspace/casadi
      tar xfz casadi-py27-np1.9.1-v2.4.3.tar.gz -C ~/Workspace/casadi

- Append location to casadi to your python path. Assuming you moved the files to
  ```~/Workspace/casadi```, add this line to your .bashrc (.bash_profile on Mac):

      export PYTHONPATH="${HOME}/Workspace/casadi:$PYTHONPATH"

### Run experiment

    python ball_catching/run.py single --strategies MPCStrategy

Note that this will take significantly longer than the other strategies due to the complexity of the method.

## Proofs

The following proofs are available in jupyter notebook:

- **Analysis of Chapman's Strategy** (Section 4.2.1): ```notebook/proofs-chapman.ipynb```
- **Angular Representation Violates Markov Property** (Section 4.2.1, Supplementary material): ```notebook/proof-angular-non-markov.ipynb```

To inspect the proofs, run jupyter notebook in the ball_catching/notebook folder and then open the notebooks in a browser

    cd ball_catching/notebook
    jupyter notebook


## References

[Belousov, 2016] Belousov B, Neumann G, Rothkopf CA, Peters J. Catching heuristics are optimal control policies. In: Advances in Neural Information Processing Systems (NIPS). Barcelona, Spain; 2016. p. 1426–1434.

This package contains adapted code from [scipy](http://www.scipy.org) and [easy_catch](https://github.com/b4be1/easy_catch)
