'''
Author: nico

Loads from a CSV file a set of parameters, where each row represents an experiment.
Each script is run in a separate screen session.
'''

import pandas as pd
import argparse, os, sys
from screener import Screener

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--experiment', help='Experiment CSV file to load.', type=str, default=None)
parser.add_argument('--condaenv', help='Conda environment to activate.', type=str, default=None)
parser.add_argument('--pythonv', help='Python version to use', type=str, default='python3')
parser.add_argument('--pythonpath', help='Python path additions', type=str, default=None)
args = parser.parse_args()

if args.experiment is not None:
    exp_filename = 'experiments/' + args.experiment + '.csv'
    experiment = pd.read_csv(exp_filename)
    cmd_base = args.pythonv + ' script/'
    if args.pythonpath:
        cmd_base = 'PYTHONPATH="PYTHONPATH:' + args.pythonpath + '" ' + cmd_base
    if args.condaenv is not None:
        cmd_base = 'source activate baselines && ' + cmd_base
    param_cols = list(experiment)
    param_cols.remove('script')
    cmds = []
    for index, row in experiment.iterrows():
        _c = cmd_base + row['script'] + ' '
        if len(param_cols) > 0:
            _c = _c + 'with '
        for p in param_cols:
            _c += str(p).strip() + '=' + str(row[p]).strip() + ' '
        cmds.append(_c)
    print(cmds)
    #scr = Screener()
    #scr.run(cmds, name=args.experiment)
else:
    print("Provide an experiment file.")
    exit(-1)
