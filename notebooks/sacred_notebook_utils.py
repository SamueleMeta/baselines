'''
    Collection of utils used for displaying sacred results in notebooks.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

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
def plot_all(my_runs, metric, axis, legend=True):
    # Extract the metric
    run_keys = list(my_runs.keys())
    metric_matrix = np.array([my_runs[key]['metrics'][metric]['values'] for key in run_keys])
    for i in range(metric_matrix.shape[0]):
        axis.plot(metric_matrix[i], label='Seed:'+str(my_runs[run_keys[i]]['config']['seed']))
    if legend:
        axis.legend()
