#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import rllab

from baselines.envs.rllab_wrappers import Rllab2GymWrapper

def rllab_env_from_name(env):
    if env == 'swimmer':
        return rllab.envs.mujoco.swimmer_env.SwimmerEnv
    elif env == 'ant':
        return rllab.envs.mujoco.ant_env.AntEnv
    elif env == 'half-cheetah':
        return rllab.envs.mujoco.half_cheetah_env.HalfCheetahEnv
    elif env == 'hopper':
        return rllab.envs.mujoco.hopper_env.HopperEnv
    elif env == 'simple-humanoid':
        return rllab.envs.mujoco.simple_humanoid_env.SimpleHumanoidEnv
    elif env == 'full-humanoid':
        return rllab.envs.mujoco.humanoid_env.HumanoidEnv
    elif env == 'walker':
        return rllab.envs.mujoco.walker2d_env.Walker2DEnv
    elif env == 'cartpole':
        return rllab.envs.box2d.cartpole_env.CartpoleEnv
    elif env == 'mountain-car':
        return rllab.envs.box2d.mountain_car_env.MountainCarEnv
    elif env == 'inverted-pendulum':
        return rllab.envs.box2d.cartpole_swingup_env.CartpoleSwingupEnv
    elif env == 'acrobot':
        return rllab.envs.box2d.double_pendulum_env.DoublePendulumEnv
    elif env == 'inverted-double-pendulum':
        return rllab.envs.mujoco.inverted_double_pendulum_env.InvertedDoublePendulumEnv

def train(env, max_iters, num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, policy, max_offline_iters, njobs=1):

    def make_env():
        env_rllab = rllab_env_from_name(env_id)()
        env_rllab = Rllab2GymWrapper(env_rllab)
        return env_rllab

    env = make_env()
    print(env.reset())
    print(env.observation_space)
    print(env.action_space)

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
