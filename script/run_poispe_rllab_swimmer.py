#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys
sys.path.append('/home/alberto/rllab')

from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.poisnpe as poisnpe
import baselines.common.tf_util as U
import numpy as np
from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.mujoco.swimmer_env import SwimmerEnv


def train(num_episodes, horizon, seed):

    sess = U.single_threaded_session()
    sess.__enter__()

    DIR = '../results/poisnpe/swimmer/seed_' + str(seed)
    gamma = 1.
    env = SwimmerEnv()
    env = Rllab2GymWrapper(env)

    rmax = None #Empirical
    
    pol_maker = lambda name: PeMlpPolicy(name,
                      env.observation_space,
                      env.action_space,
                      hid_layers=[],
                      diagonal=True,
                      use_bias=False,
                      standardize_input=False,
                      seed=seed)
    
    poisnpe.learn(env,
              pol_maker,
              gamma=gamma,
              batch_size=100,
              task_horizon=horizon,
              max_iterations=500,
              save_to=DIR,
              verbose=1,
              feature_fun=np.ravel,
              correct_ess=True,
              normalize=True,
              max_offline_ite=100,
              max_search_ite=30,
              bound_name='z',
              rmax=rmax)

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    args = parser.parse_args()

    train(num_episodes=args.num_episodes,
          horizon=args.horizon,
          seed=args.seed)


if __name__ == '__main__':
    main()
