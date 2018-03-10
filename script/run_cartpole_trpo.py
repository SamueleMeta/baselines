#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
from baselines.envs.continuous_cartpole import CartPoleEnv
import baselines.common.tf_util as U


def train(num_episodes, horizon, seed):
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = CartPoleEnv()
    env.horizon = horizon

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                         hid_size=0, num_hid_layers=0, gaussian_fixed_var=True, use_bias=False)

    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, task_horizon=horizon, batch_size=num_episodes, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_iters=20, gamma=.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=200)
    args = parser.parse_args()
    logger.configure(dir='.', format_strs=['stdout', 'csv'])
    train(num_episodes=args.num_episodes, horizon=args.horizon, seed=args.seed)


if __name__ == '__main__':
    main()