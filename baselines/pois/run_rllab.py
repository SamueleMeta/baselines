#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import time

from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def rllab_env_from_name(env):
    if env == 'swimmer':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        return SwimmerEnv
    elif env == 'ant':
        from rllab.envs.mujoco.ant_env import AntEnv
        return AntEnv
    elif env == 'half-cheetah':
        from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
        return HalfCheetahEnv
    elif env == 'hopper':
        from rllab.envs.mujoco.hopper_env import HopperEnv
        return HopperEnv
    elif env == 'simple-humanoid':
        from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
        return SimpleHumanoidEnv
    elif env == 'full-humanoid':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        return HumanoidEnv
    elif env == 'walker':
        from rllab.envs.mujoco.walker2d_env import Walker2DEnv
        return Walker2DEnv
    elif env == 'cartpole':
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        return CartpoleEnv
    elif env == 'mountain-car':
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        return MountainCarEnv
    elif env == 'inverted-pendulum':
        from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv as InvertedPendulumEnv
        return InvertedPendulumEnv
    elif env == 'acrobot':
        from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv as AcrobotEnv
        return AcrobotEnv
    elif env == 'inverted-double-pendulum':
        from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
        return InvertedPendulumEnv
    else:
        raise Exception('Unrecognized rllab environment.')

def train(env, max_iters, num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, policy, max_offline_iters, njobs=1):

    rllab_env_class = rllab_env_from_name(env)

    def make_env(seed=0):
        env_rllab = Rllab2GymWrapper(rllab_env_class())
        return env_rllab.seed(seed)

    parallel_env = SubprocVecEnv([make_env(i + seed) for i in range(njobs)])
    print(parallel_env.reset())
    print(parallel_env.observation_space)
    print(parallel_env.action_space)
    print(parallel_env.step_async(parallel_env.action_space.sample()))
    print(parallel_env.step_wait())

    '''
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)
    '''

def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=500)
    args = parser.parse_args()
    # Configure logging
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='logs', format_strs=['stdout', 'csv'], file_name=file_name)
    # Start training
    train(env=args.env,
          max_iters=args.max_iters,
          num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          delta=args.delta,
          seed=args.seed,
          policy=args.policy,
          max_offline_iters=args.max_offline_iters,
          njobs=args.njobs)


if __name__ == '__main__':
    main()
