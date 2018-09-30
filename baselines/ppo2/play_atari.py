#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import multiprocessing
import tensorflow as tf
from time import sleep
import numpy as np

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--checkpoint', help='Checkpoint filename', default=None)
    args = parser.parse_args()

    EPISODES = 10

    env = make_atari(args.env)
    env = wrap_deepmind(env, frame_stack=4)

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    m = ppo2.Model(policy=CnnPolicy, ob_space=env.observation_space, ac_space=env.action_space,
                    nbatch_act=1, nbatch_train=1, nsteps=500, ent_coef=.01, vf_coef=.01, max_grad_norm=.01)
    m.load(args.checkpoint)


    for i in range(EPISODES):
        obs = np.array(env.reset())
        done = False
        while not done:
            a, _, _, _ = m.step(np.reshape(obs, (1,)+obs.shape))
            obs, r, done, _ = env.step(a[0])
            obs = np.array(obs)
            env.render()
            sleep(0.01)

    env.close()

if __name__ == '__main__':
    main()
