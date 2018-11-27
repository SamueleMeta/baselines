# Baselines with POIS

This repository contains the implementation of the [POIS algorithm](https://arxiv.org/abs/1809.06098).
It is based on the OpenAI baselines implementation, and uses the same implementation backbone.
We are working on synchronising this repository to the current version of OpenAI baselines.

## What's new
We provide 3 different flavours of the POIS algorithm:
- **POIS**: control-based POIS (cpu optimized, should be used in simple environments with simple policies)
- **POIS2**: control-based POIS (gpu optimized, used in complex environments or complex policies)
- **PBPOIS**: parameter-based POIS

This repository also contains a version of the algorithm controlled using [sacred](https://sacred.readthedocs.io/en/latest/). The POIS algorithms can be run using a single script, both from gym and rllab. To use rllab environments, use the prefix "*rllab.*". You should always set the *SACRED_RUNS_DIRECTORY* env variable to tell sacred where to log the run. For example:

```bash
export SACRED_RUNS_DIRECTORY=sacred_runs
python baselines/pois/run_sacred.py with env=rllab.swimmer
```
will run the POIS algorithm over the RLLAB swimmer environment logging into the *sacred_runs* directory.

We also provide an [*experiment_manager*](baselines/experiment_manager.py) script, which can be used to launch multiple runs of the algorithm in different screen and to manage them, using a CSV file to specify the parameters of each run. An example CSV is contained in the [experiments](experiments) directory. To run the sample:

```bash
python baselines/experiment_manager.py --command=launch --dir=experiments --name=sample_experiment --sacred --sacred_dir=sacred_runs
```

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
