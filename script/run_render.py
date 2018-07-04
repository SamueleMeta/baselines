'''
    Runs and renders an environment with a trained policy.
'''
import sys

from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois
import baselines.common.tf_util as U
import time
import os
import pickle
import tensorflow as tf

sys.path.append('../rllab')
from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from baselines.envs.rllab_wrappers import Rllab2GymWrapper

from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv

from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv as InvertedPendulumEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv as AcrobotEnv


def play(env, num_episodes, horizon, seed, policy, weights_file):

    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    def make_policy(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=tf.contrib.layers.xavier_initializer(),
                         output_W_init=tf.contrib.layers.xavier_initializer())

    sess = U.make_session()
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)

    if env == 'swimmer':
            make_env_rllab = SwimmerEnv
    elif env == 'ant':
        make_env_rllab = AntEnv
    elif env == 'half-cheetah':
        make_env_rllab = HalfCheetahEnv
    elif env == 'hopper':
        make_env_rllab = HopperEnv
    elif env == 'simple-humanoid':
        make_env_rllab = SimpleHumanoidEnv
    elif env == 'full-humanoid':
        make_env_rllab = HumanoidEnv
    elif env == 'walker':
        make_env_rllab = Walker2DEnv
    elif env == 'cartpole':
        make_env_rllab = CartpoleEnv
    elif env == 'mountain-car':
        make_env_rllab = MountainCarEnv
    elif env == 'inverted-pendulum':
        make_env_rllab = InvertedPendulumEnv
    elif env == 'acrobot':
        make_env_rllab = AcrobotEnv
    elif env == 'inverted-double-pendulum':
        make_env_rllab = InvertedDoublePendulumEnv

    env_rllab = make_env_rllab()
    env = Rllab2GymWrapper(env_rllab)

    #env = gym.make('Swimmer-v2')

    # Load weights and make policy
    file = open(weights_file, 'rb')
    theta = pickle.load(file)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = make_policy('pi', ob_space, ac_space)
    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]
    set_parameter = U.SetFromFlat(var_list)
    set_parameter(theta)

    obs = env.reset()
    done = False
    while not done:
        ac, vpred = pi.act(True, obs)
        obs, r, done, _ = env.step(ac)
        env_rllab.render()

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # Load weights from file
    play(args.env,
        args.num_episodes,
        args.horizon,
        args.seed,
        args.policy,
        args.checkpoint)

if __name__ == '__main__':
    main()
