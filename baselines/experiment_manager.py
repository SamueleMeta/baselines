'''
    This script helps creating and managing experiments.
    Possible commands:
    - launch: launch an experiment loading its specification from a CSV file
    - view: list the experiments which are still running
    - stop: stop all the runners of the experiment
'''

import pandas as pd
import argparse, os, sys, re
from multiprocessing import Pool
from screenutils import Screen, list_screens
from datetime import datetime

class Screener(object):

    def command_sender(self, zipped_pair):
        screen, command = zipped_pair
        screen.send_commands(command)

    def run(self, commands, name='s'):
        n_screens = len(commands)
        screens = [Screen(name+'_%d' % (i+1), True) for i in range(n_screens)]

        p = Pool(n_screens)
        p.map(self.command_sender, zip(screens, commands))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--command', help='Command to execute.', type=str, default='launch', choices=['launch', 'view', 'stop'])
# Experiment selection
parser.add_argument('--name', help='Name of the experiment', type=str, default=None)
parser.add_argument('--dir', help='Directory from which to load the experiment (to launch).', type=str, default=None)
# Env
parser.add_argument('--condaenv', help='Conda environment to activate.', type=str, default=None)
parser.add_argument('--pythonv', help='Python version to use', type=str, default='python3')
parser.add_argument('--pythonpath', help='Pythonpath to use for script.', type=str, default=None)
parser.add_argument('--cuda_devices', help='CUDA visible devices.', type=str, default='')
# Sacred
parser.add_argument('--sacred', action='store_true', default=False, help='Enable sacred.')
parser.add_argument('--sacred_dir', help='Dir used by sacred to log.', type=str, default=None)
parser.add_argument('--sacred_slack', help='Config file for slack.', type=str, default=None)
parser.add_argument('--dirty', action='store_true', default=False, help='Enable sacred dirty running.')
args = parser.parse_args()

if args.command == 'launch':
    assert args.name is not None, "Provide an experiment name."
    assert args.dir is not None, "Provide a directory to load the experiment."
    # Load experiment
    experiment_path = args.dir + '/' + args.name + '.csv'
    experiment = pd.read_csv(experiment_path)
    # Start build base command
    cmd_base = ''
    # Set env variables
    cmd_base += 'export CUDA_VISIBLE_DEVICES=' + args.cuda_devices + ' && '
    cmd_base += 'export EXPERIMENT_NAME=' + args.name + ' && '
    if args.sacred_dir and args.sacred:
        cmd_base += 'export SACRED_RUNS_DIRECTORY=' + args.sacred_dir + ' && '
    if args.sacred_slack and args.sacred:
        cmd_base += 'export SACRED_SLACK_CONFIG=' + args.sacred_slack + ' && '
    if args.pythonpath:
        cmd_base += "export PYTHONPATH='PYTHONPATH:" + args.pythonpath + "' && "
    if args.condaenv:
        cmd_base += 'source activate ' + args.condaenv + ' && '
    # Parse the CSV
    param_cols = list(experiment)
    param_cols.remove('script')
    # Build the commands
    cmd_base += args.pythonv + ' '
    cmds = []
    for index, row in experiment.iterrows():
        # Get the script, check if we need to use sacred (just append _sacred to script name)
        script = row['script']
        if args.sacred:
            script += '_sacred'
        script = 'baselines/' + script + '.py '
        _c = cmd_base + script
        # Check if dirty and if to use with
        if args.sacred and not args.dirty:
            _c += '-e '
        if args.sacred and len(param_cols) > 0:
            _c += 'with '
        # Add experiment_name to params
        if args.sacred:
            _c += 'experiment_name=' + args.name + ' '
        else:
            _c += '--experiment_name=' + args.name + ' '
        # Params
        for p in param_cols:
            if args.sacred:
                _c += str(p).strip() + '=' + str(row[p]).strip() + ' '
            else:
                _c += '--' + str(p).strip() + '=' + str(row[p]).strip() + ' '
        # Add the exit command to terminate the experiment
        _c += '&& exit'
        cmds.append(_c)
    scr = Screener()
    scr.run(cmds, name=args.name)

elif args.command == 'view':
    from baselines.common.sacred_utils import load_runs, filter_runs
    from baselines.common import colorize
    assert args.name is not None, "Provide an experiment name."
    assert args.dir is not None, "Provide a directory for experiment."
    rule = re.compile(args.name + '_*')
    # Get all screens
    all_active_screens = 0
    for s in list_screens():
        if rule.match(s.name):
            all_active_screens += 1
    # Load runs to get active ones
    runs = load_runs(args.dir)
    running_runs = filter_runs({'run.status': 'RUNNING'}, runs)
    print(colorize("==========================================", color='red'))
    max_eta, max_duration = None, None
    for key in running_runs.keys():
        run = running_runs[key]
        print(colorize('Run:', color='blue'), "{0} ({1})".format(key, run['config']['env']))
        print("\t" + colorize("Steps:", color='blue') +
                "{0}/{1}".format(len(run['metrics']['EpRewMean']['steps'])+1, run['config']['max_iters']) +
                "\t\t" + colorize("Reward:", color='blue') + "{0}".format(run['metrics']['EpRewMean']['values'][-1]) +
                "\t\t" + colorize("Seed:", color='blue') + "{0}".format(run['config']['seed']) +
                "\t\t" + colorize("Delta:", color='blue') + "{0}".format(run['config']['delta']))
        completion = (len(run['metrics']['EpRewMean']['steps'])+1) / run['config']['max_iters']
        start_time = datetime.strptime(run['run']['start_time'], '%Y-%m-%dT%H:%M:%S.%f')
        duration = datetime.utcnow() - start_time
        eta = duration * (1 - completion) / completion
        max_eta = max(eta, max_eta) if max_eta is not None else eta
        max_duration = max(duration, max_duration) if max_duration is not None else duration
    if len(running_runs.keys()) == 0:
        print(colorize("Done.", color='red'))
    else:
        t = max_eta.total_seconds()
        d = max_duration.total_seconds()
        print(colorize("==========================================", color='red'))
        print(colorize("Active screens: {0}".format(all_active_screens), color='red'))
        print(colorize("Active runs: {0}".format(len(running_runs.keys())), color='red'))
        print(colorize("Elapsed time: {0} hours, {1} minutes, {2} seconds".format(int(d // 3600), int((d%3600)//60), int(d%3600)%60), color='red'))
        print(colorize("ETA: {0} hours, {1} minutes, {2} seconds".format(int(t // 3600), int((t%3600)//60), int(t%3600)%60), color='red'))
    print(colorize("==========================================", color='red'))

elif args.command == 'stop':
    assert args.name is not None, "Provide an experiment name."
    rule = re.compile(args.name + '_*')
    # Get all screens
    for s in list_screens():
        if rule.match(s.name):
            print("Stopping", s.name)
            s.kill()

else:
    raise Exception('Unrecognized command.')
