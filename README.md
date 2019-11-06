# Baselines with POIS

This repository contains the implementation of the [POIS algorithm](https://arxiv.org/abs/1809.06098).
It is based on the OpenAI baselines implementation, and uses the same implementation backbone.
We are working on synchronising this repository to the current version of OpenAI baselines.

## What's new
We provide 3 different flavours of the POIS algorithm:
- **POIS**: control-based POIS (cpu optimized, should be used in simple environments with simple policies)
- **POIS2**: control-based POIS (gpu optimized, used in complex environments or complex policies)
- **PBPOIS**: parameter-based POIS

This repository also contains a version of the algorithm controlled using [sacred](https://sacred.readthedocs.io/en/latest/). More info on how to launch experiments with sacred are available in the section *Launching experiments* of this readme.

## Installation
You can install it by typing:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

## Available algorithms
- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [PB-POIS](baselines/pbpois)
- [POIS](baselines/pois)
- [POIS2](baselines/pois2)
- [PPO1](baselines/ppo1) (Multi-CPU using MPI)
- [PPO2](baselines/ppo2) (Optimized for GPU)
- [TRPO](baselines/trpo_mpi)

## Citing
To cite the OpeanAI baselines repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }

To cite the POIS paper:

    @misc{metelli2018policy,
      title={Policy Optimization via Importance Sampling},
      author={Alberto Maria Metelli and Matteo Papini and Francesco Faccio and Marcello Restelli},
      year={2018},
      eprint={1809.06098},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

## Launching experiments

In this framework, we allow launching experiments, i.e. a collection of runs with specific parameters. Run parameters are specified in CSV files and placed inside the `experiments` directory. See that directory for an example of the content.
Launching the experiment is done by calling:
```python3 baselines/experiment_manager --command=launch --dir=experiments --name=mujoco```
This command launches an experiment named *mujoco*, searching for the relative `experiments/mujoco.csv` file. 

Running the experiment is a completely parallelized task, so multiple runs are spread over multiple processors. This is done by instantiating one virtual screen for each of the single runs. The `experiment_manager`, when launching an experiment, creates one screen for every run and issues to it a script to run with all the requested parameters (as specified in the experiment CSV).

The `experiment_manager` script can be also used to print statistics or even to stop a particular experiment, making it easier to manage a lot of virtual screens simultaneously.

Inside the 4 POIS directories, namely:
- `baselines/pois`: single policy POIS, decentralized policies for each worker
- `baselines/pois2`: single policy POIS, centralized policy for every worker
- `baselines/pomis`: multiple POIS, decentralized policies for each worker
- `baselines/pomis2`: multiple POIS, centralized policy for every worker, 

we can find 2 type of scripts to launch single runs:
- `run.py`: run with basic logging, i.e. tensorboard and csv
- `run_sacred.py`: run with sacred logging.

All the experiments contained in the paper are logged using sacred, so we can recover the exact parameters and exact code that generated that result. To enable sacred using the `experiment_manager`, add the keywords `--sacred --sacred_dir=sacred_runs`; in this way, the output of sacred will be saved inside `sacred_runs`.

The script `script/sacred_manager` contains many utility functions to manage these logging directories, e.g. for deleting or selecting runs.

## Using RL-LAB

To use rllab environments, use the prefix "*rllab.*" in the environment name. A list of all the environments can be found [here](baselines/common/rllab_utils.py).