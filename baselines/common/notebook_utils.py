#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:57:26 2018

@author: matteo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts


def read_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    iterations = len(df)
    df['CumAvgRet'] = np.cumsum(df['AvgRet'])/iterations
    return df

def moments(dfs):
    concat_df = pd.concat(dfs, axis=1)
    mean_df = pd.concat(dfs, axis=1).groupby(by=concat_df.columns, axis=1).mean()
    std_df = pd.concat(dfs, axis=1).groupby(by=concat_df.columns, axis=1).std()
    return mean_df, std_df

def compare(means, stds, conf, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for mean, std in zip(means, stds):
        n_runs = len(mean)
        ax.plot(mean.index, mean)
        interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/np.sqrt(n_runs))
        ax.fill_between(mean.index, interval[0], interval[1], alpha=0.3)
    if ylim: ax.set_ylim(ylim)
    return fig

def plot_all(dfs, key='AvgRet', ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for df in dfs:
        value = df[key]
        ax.plot(value.index, value)
    return fig

def plot_ci(dfs, conf=0.95, key='AvgRet', ylim=None):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key]
    std = std_df[key]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mean.index, mean)
    interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/np.sqrt(n_runs))
    ax.fill_between(mean.index, interval[0], interval[1], alpha=0.3)
    if ylim: ax.set_ylim(ylim)
    return fig

def plot_data(path, key='VanillaAvgRet'):
    df = pd.read_csv(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean = df[key]
    ax.plot(mean.index, mean)
    return fig

def print_ci(dfs, conf=0.95, key='CumAvgRet'):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key][len(mean_df)-1]
    std = std_df[key][len(mean_df)-1]
    interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/np.sqrt(n_runs))
    print('%f\t[%f, %f]' % (mean, interval[0], interval[1]))