#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This script runs the RLLab implementation of TRPO on various environments.
    The environments, in this case, are not wrapped for gym.
'''

# Common imports
import sys, re, os, time, logging
from collections import defaultdict
# RLLab
import rllab
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.core.network import MLP
# Lasagne
import lasagne.nonlinearities as NL
import lasagne.init as LI
# Baselines
from baselines import logger
from baselines.common.rllab_utils import rllab_env_from_name
from baselines.common.cmd_util import get_env_type

def train(env, policy, policy_init, num_episodes, horizon, **alg_args):

    # Getting the environment
    env_class = rllab_env_from_name(env)
    env = normalize(env_class())

    # Policy initialization
    if policy_init == 'zeros':
        initializer = LI.Constant(0)
    elif policy_init == 'normal':
        initializer = LI.Normal()
    else:
        raise Exception('Unrecognized policy initialization.')

    # Creating the policy
    if policy == 'linear':
        obs_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim
        mean_network = MLP(
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=tuple(),
                    hidden_nonlinearity=NL.tanh,
                    output_nonlinearity=None,
                    output_b_init=None,
                    output_W_init=initializer,
                )
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=tuple(),
            mean_network=mean_network,
            log_weights=False,
        )
    else:
        raise Exception('NOT IMPLEMENTED.')

    # Creating baseline
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # Adding max_episodes constraint. If -1, this is unbounded
    if num_episodes > 0:
        alg_args['max_episodes'] = num_episodes

    # Run algorithm
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=horizon * num_episodes,
        whole_paths=True,
        max_path_length=horizon,
        **alg_args
    )
    algo.train()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--policy_init', type=str, default='xavier')
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--experiment_name', type=str, default='none')
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = '%s_TRPO_sz=%s_seed=%s_%s' % (args.env.upper(), args.step_size, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)
    train(env=args.env,
          policy=args.policy,
          policy_init=args.policy_init,
          num_episodes=args.num_episodes,
          horizon=args.horizon,
          seed=args.seed,
          n_itr=args.max_iters,
          step_size=args.step_size,
          discount=args.gamma,)

if __name__ == '__main__':
    main()
