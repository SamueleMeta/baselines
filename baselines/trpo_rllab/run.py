#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This script runs the RLLab implementation of TRPO on various environments.
    The environments, in this case, are not wrapped for gym.
'''

# Common imports
import sys, re, os, time, logging
from collections import defaultdict
# RLLab
import rllab
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# Baselines
from baselines.common.rllab_utils import rllab_env_from_name
from baselines.common.cmd_util import get_env_type
