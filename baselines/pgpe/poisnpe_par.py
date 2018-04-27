#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:13:18 2018

@author: matteo
"""
"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
"""

import numpy as np
from baselines import logger
import warnings
from contextlib import contextmanager
import time
from baselines.common import colorize
from joblib import Parallel, delayed
import random

N_JOBS = 4

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))

def eval_trajectory(env, pol, gamma, task_horizon, feature_fun, seed):
    pol.seed(seed)
    env.seed(seed)
    theta = pol.resample()
    ret = disc_ret = 0
    max_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t<task_horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma**t * r
        max_ret = max(abs(r), max_ret)
        t+=1
        
    return ret, disc_ret, t, max_ret, theta

#BINARY line search
def line_search(pol, newpol, actor_params, rets, alpha, natgrad, correct=True, normalize=True, max_search_ite=30,
                bound_name='t', rmax=None):
    rho_init = newpol.eval_params()
    low = 0.
    high = None
    bound_init = newpol.eval_bound(actor_params, rets, behavioral=pol, correct=correct, normalize=normalize, 
                                   bound_name=bound_name, rmax=rmax)
    #old_delta_bound = 0.
    rho_opt = rho_init
    i_opt = 0.
    delta_bound_opt = 0.
    epsilon_opt = 0.
    epsilon = 1.
    
    for i in range(max_search_ite):
        rho = rho_init + epsilon * natgrad * alpha
        newpol.set_params(rho)
        bound = newpol.eval_bound(actor_params, rets, behavioral=pol, correct=correct, normalize=normalize, 
                                  bound_name=bound_name, rmax=rmax)
        delta_bound = bound - bound_init        
        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            rho_opt = rho
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        old_epsilon = epsilon
        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.
        if abs(old_epsilon - epsilon) < 1e-6:
            break
    
    return rho_opt, epsilon_opt, delta_bound_opt, i_opt+1

def optimize_offline(pol, newpol, actor_params, rets, grad_tol=1e-4, bound_tol=1e-4, max_offline_ite=100, 
                     correct_ess=True, normalize=True, max_search_ite=30,
                     stop_sigma=False, bound_name='t', rmax=None):
    improvement = 0.
    rho = pol.eval_params()
    
    
    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", 
                      "gradient norm", "delta bound ite", "delta bound tot"))
    
    for i in range(max_offline_ite):
        #Candidate policy
        newpol.set_params(rho)
        
        #Bound with gradient
        bound, grad = newpol.eval_bound_and_grad(actor_params, rets, behavioral=pol, correct=correct_ess, normalize=normalize,
                                                 bound_name=bound_name, rmax=rmax)
        if np.any(np.isnan(grad)):
            warnings.warn('Got NaN gradient! Stopping!')
            return rho, improvement
        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            return rho, improvement     
        
        #Natural gradient
        if newpol.diagonal: 
            natgrad = grad/(newpol.eval_fisher() + 1e-24)
        else:
            raise NotImplementedError
        assert np.dot(grad, natgrad) >= 0
        if i>0 and newpol.diagonal and stop_sigma:
            natgrad[len(natgrad)//2:] = 0
        grad_norm = np.sqrt(np.dot(grad, natgrad))
        if grad_norm < grad_tol:
            print("stopping - gradient norm < gradient_tol")
            print(rho)
            return rho, improvement
        
        #Step size search
        alpha = 1. / grad_norm ** 2
        rho, epsilon, delta_bound, num_line_search = line_search(pol, 
                                                                 newpol, 
                                                                 actor_params, 
                                                                 rets, 
                                                                 alpha, 
                                                                 natgrad, 
                                                                 correct=correct_ess,
                                                                 normalize=normalize,
                                                                 max_search_ite=max_search_ite,
                                                                 bound_name=bound_name,
                                                                 rmax=rmax)
        newpol.set_params(rho)
        improvement+=delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, grad_norm, delta_bound, improvement))
        if delta_bound < bound_tol:
            print("stopping - delta bound < bound_tol")
            print(rho)
            return rho, improvement
    
    return rho, improvement


def learn(env, pol_maker, gamma, batch_size, task_horizon, max_iterations, 
          feature_fun=None, max_offline_ite=100, correct_ess=True, normalize=True,
          max_search_ite=30, stop_sigma=False, bound_name='t', rmax=None,
          verbose=True, 
          save_to=None):
    
    #Logging
    format_strs = []
    if verbose: format_strs.append('stdout')
    if save_to: format_strs.append('csv')
    logger.configure(dir=save_to, format_strs=format_strs)

    pol = pol_maker('pol')
    newpol = pol_maker('oldpol')
    newpol.set_params(pol.eval_params())
    
    #Learning iteration
    for it in range(max_iterations):
        logger.log('\n********** Iteration %i ************' % it)
        rho = pol.eval_params() #Higher-order-policy parameters
        if verbose>1:
            logger.log('Higher-order parameters: ', rho)
            print(len(rho))
        if save_to: np.save(save_to + '/weights_' + str(it), rho)
            
        #Batch of episodes
        #TODO: try symmetric sampling
        with timed('Sampling'):
            frozen_pol = pol.freeze()
            traj_seeds = [random.randint(0,10**5) for _ in range(batch_size)]
            rets, disc_rets, ep_lens, max_rets, actor_params = map(list, zip(*Parallel(n_jobs=N_JOBS)(delayed(eval_trajectory)(env, frozen_pol, gamma, task_horizon, feature_fun, traj_seed)
                for traj_seed in traj_seeds)))
            
        logger.log('Performance: ', np.mean(rets))
        #if save_to: np.save(save_to + '/rets_' + str(it), rets)
            
        
        #Offline optimization
        with timed('Optimizing offline'):
            if rmax is None:
                _rmax = sum(max(max_rets)*gamma**i for i in range(task_horizon)) 
            else:
                _rmax = rmax
                if verbose: print('Using empirical maxRet %f' % _rmax)
            rho, improvement = optimize_offline(pol, newpol, actor_params, rets, correct_ess=correct_ess,
                                                normalize=normalize,
                                                max_offline_ite=100,
                                                max_search_ite=max_search_ite,
                                                stop_sigma=stop_sigma,
                                                bound_name=bound_name,
                                                rmax=_rmax)
            newpol.set_params(rho)
            assert(improvement>=0.)
        
        logger.log('Recap of iteration %i' % it)
        unn_iws = newpol.eval_iws(actor_params, behavioral=pol, normalize=False)
        iws = unn_iws/np.sum(unn_iws)
        ess = np.linalg.norm(unn_iws, 1) ** 2 / np.linalg.norm(unn_iws, 2) ** 2
        J, varJ = newpol.eval_performance(actor_params, disc_rets, behavioral=pol)
        eRenyi = np.exp(newpol.eval_renyi(pol))
        
        logger.record_tabular('Bound', newpol.eval_bound(actor_params, rets, behavioral=pol, correct=correct_ess, 
                                                         normalize=normalize, bound_name=bound_name, rmax=_rmax))
        logger.record_tabular('ESSClassic', ess)
        logger.record_tabular('ESSRenyi', batch_size/eRenyi)
        logger.record_tabular('MaxVanillaIw', np.max(unn_iws))
        logger.record_tabular('MinVanillaIw', np.min(unn_iws))
        logger.record_tabular('AvgVanillaIw', np.mean(unn_iws))
        logger.record_tabular('VarVanillaIw', np.var(unn_iws, ddof=1))
        logger.record_tabular('MaxNormIw', np.max(iws))
        logger.record_tabular('MinNormIw', np.min(iws))
        logger.record_tabular('AvgNormIw', np.mean(iws))
        logger.record_tabular('VarNormIw', np.var(iws, ddof=1))
        logger.record_tabular('eRenyi2', eRenyi)
        logger.record_tabular('AvgRet', np.mean(rets))
        logger.record_tabular('VarRet', np.var(rets, ddof=1))
        logger.record_tabular('VarDiscRet', np.var(disc_rets, ddof=1))
        logger.record_tabular('AvgDiscRet', np.mean(disc_rets))
        logger.record_tabular('J', J)
        logger.record_tabular('VarJ', varJ)
        logger.record_tabular('BatchSize', batch_size)
        logger.record_tabular('AvgEpLen', np.mean(ep_lens))
        logger.dump_tabular()
        
        
        #Update behavioral
        pol.set_params(newpol.eval_params()) 