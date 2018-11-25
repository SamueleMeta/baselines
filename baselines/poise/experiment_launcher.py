from baselines.poise import run
#
# import sys
# import os
# import numpy as np
# import warnings
# import time
# import random
from joblib import Parallel, delayed
# sys.path.append('..')


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='LQG1D-v0')
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--delta', type=float, default=0.3)#delta piccolo -> grande bonus
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--render_after', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--bounded_policy', type=str, default='True')
    args = parser.parse_args()
    special_args = []
    for i in frange(0.2, 0.8, 0.2):
        for j in range(3):
            special_args.append([str(i), str(j)])

    n_jobs = len(special_args)
    Parallel(n_jobs=n_jobs)(delayed(run.main)(
        ['--delta', special_args[i][0],
         '--seed', special_args[i][1],
         '--env', args.env,
         '--horizon', args.horizon,
         '--iw_norm', args.iw_norm,
         '--file_name', args.file_name,
         ]
        ) for i in range(n_jobs))
