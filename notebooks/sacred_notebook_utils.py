'''
    Collection of utils used for displaying sacred results in notebooks.
'''

import numpy as np
import glob, sys, json, re
import matplotlib.pyplot as plt
import scipy.stats as sts

'''
    This function loads all the available runs inside the specified directory.
'''
def load_runs(base_directory):
    if base_directory[-1] != '/':
        base_directory += '/'
    runs = {}
    runs_filenames = glob.glob(base_directory + '*/config.json')
    run_extractor = re.compile(base_directory + '([0-9]+)/config.json')
    for r in runs_filenames:
        try:
            run_number = int(run_extractor.match(r).group(1))
            runs[run_number] = {}
            runs[run_number]['config'] = json.load(open(base_directory + str(run_number) + '/config.json'))
            runs[run_number]['run'] = json.load(open(base_directory + str(run_number) + '/run.json'))
            runs[run_number]['metrics'] = json.load(open(base_directory + str(run_number) + '/metrics.json'))
        except:
            del runs[run_number]
    return runs

'''
    This function returns a filtered dictionary containing all the runs that
    meet the requirements specified with the query parameter. Parameters at
    different levels of the run specification are given using the dot notation.
'''
def filter_runs(query, runs):
    keys = list(runs.keys())
    for key, value in query.items():
        # Check if the still unfiltered runs have the specified parameter. If not, remove them
        _keys = []
        for run_key in keys:
            obj = runs[run_key]
            # Dot notation at any level
            for qk in key.split('.'):
                obj = obj[qk]
            # Check if it matches value
            if obj == value:
                _keys.append(run_key)
        keys = _keys
    # Now create a filtered object only with the selected runs
    _runs = {key: runs[key] for key in keys}
    return _runs

'''
    This function creates a plot in the given axis. This plot shows, for a certain
    specified metric, the mean and the confidence interval of the metric for the
    given runs.
'''
def plot_mean_ci(my_runs, metric, axis, conf=0.95):
    # Extract the metric
    metric_matrix = np.array([value['metrics'][metric]['values'] for key, value in my_runs.items()])
    # Get mean and ci
    mean = np.mean(metric_matrix, axis=0)
    std = np.std(metric_matrix, axis=0)
    interval = sts.t.interval(conf, metric_matrix.shape[0]-1, loc=mean, scale=std/np.sqrt(metric_matrix.shape[0]))
    axis.plot(mean)
    axis.fill_between(range(len(mean)),interval[0], interval[1], alpha=0.3)

'''
    This function creates a plot of all the metrics specified in the given runs.
'''
def plot_all(my_runs, metric, axis):
    # Extract the metric
    metric_matrix = np.array([value['metrics'][metric]['values'] for key, value in my_runs.items()])
    for i in range(metric_matrix.shape[0]):
        axis.plot(metric_matrix[i])
