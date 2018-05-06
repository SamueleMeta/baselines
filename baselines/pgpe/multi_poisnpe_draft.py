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

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))

def eval_trajectory(env, pol, gamma, task_horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t<task_horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma**t * r
        t+=1
        
    return ret, disc_ret, t

#BINARY line search
def line_search_binary(pol, newpol, actor_params, rets, alpha, natgrad, 
                normalize=True,
                use_rmax=True,
                use_renyi=True,
                max_search_ite=30, rmax=None, delta=0.2):
    rho_init = newpol.eval_params()
    low = 0.
    high = None
    bound_init = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
    #old_delta_bound = 0.
    rho_opt = rho_init
    i_opt = 0.
    delta_bound_opt = 0.
    epsilon_opt = 0.
    epsilon = 1.
    
    for i in range(max_search_ite):
        rho = rho_init + epsilon * natgrad * alpha
        newpol.set_params(rho)
        bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
        delta_bound = bound - bound_init        
        if (np.isnan(bound)).any():
            warnings.warn('Got NaN bound value: rolling back!')
            high = epsilon
        elif np.amax(delta_bound) <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            rho_opt = rho
            delta_bound_opt = np.amax(delta_bound)
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

def line_search_parabola(pol, newpol, actor_params, rets, alpha, natgrad, 
                normalize=True,
                use_rmax=True,
                use_renyi=True,
                max_search_ite=30, rmax=None, delta=0.2):
    epsilon = 1.
    epsilon_old = 0.
    max_increase=2. 
    delta_bound_tol=1e-4
    delta_bound_old = -np.inf
    bound_init = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
    rho_old = rho_init = newpol.eval_params()

    for i in range(max_search_ite):

        rho = rho_init + epsilon * alpha * natgrad
        newpol.set_params(rho)

        bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return rho_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        
        if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
            epsilon = epsilon_old * max_increase
        else:
            epsilon = epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))
        
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return rho_init, 0., 0., i+1
            else:
                return rho_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        rho_old = rho

    return rho_old, epsilon_old, delta_bound_old, i+1

def optimize_offline(pol, newpol, actor_params, rets, grad_tol=1e-4, bound_tol=1e-4, max_offline_ite=100, 
                     normalize=True, 
                     use_rmax=True,
                     use_renyi=True,
                     max_search_ite=30,
                     rmax=None, delta=0.2, use_parabola=False):
    improvement = 0.
    rho = pol.eval_params()
    
    
    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", 
                      "gradient norm", "delta bound ite", "delta bound tot"))
    
    natgrad = None
    
    for i in range(max_offline_ite):
        #Candidate policy
        newpol.set_params(rho)
        
        #Bound with gradient
        bound, grad = newpol.eval_bound_and_grad(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
        if np.any(np.isnan(grad)):
            warnings.warn('Got NaN gradient! Stopping!')
            return rho, improvement
        if (np.isnan(bound)).any():
            warnings.warn('Got NaN bound! Stopping!')
            return rho, improvement     

            
        #Natural gradient
        fisher = newpol.eval_fisher()
        natgrad = grad/(fisher + 1e-24)
        assert np.dot(grad, natgrad) >= 0
        
        #Step size search
        alpha = np.zeros(len(grad))
        max_grad_norm = -np.inf
        for i in range(len(grad)//2):    
            grad_norm = grad[i]*natgrad[i] + grad[2*i]*natgrad[2*i]
            max_grad_norm = max(grad_norm, max_grad_norm)
            alpha[i] = 1. / grad_norm**2
            alpha[2*i] = 1./grad_norm**2
        
        if max_grad_norm < grad_tol:
            print("stopping - max_gradient norm < gradient_tol")
            return rho, improvement

        line_search = line_search_parabola if use_parabola else line_search_binary
        rho, epsilon, delta_bound, num_line_search = line_search(pol, 
                                                                 newpol, 
                                                                 actor_params, 
                                                                 rets, 
                                                                 alpha, 
                                                                 natgrad, 
                                                                 normalize=normalize,
                                                                 use_rmax=use_rmax,
                                                                 use_renyi=use_renyi,
                                                                 max_search_ite=max_search_ite,
                                                                 rmax=rmax,
                                                                 delta=delta)
        newpol.set_params(rho)
        improvement+=delta_bound
        print(fmtstr % (i+1, epsilon, np.amax(alpha*epsilon), num_line_search, grad_norm, delta_bound, improvement))
        if delta_bound < bound_tol:
            print("stopping - delta bound < bound_tol")
            return rho, improvement
    
    return rho, improvement


def learn(env, pol_maker, gamma, batch_size, task_horizon, max_iterations, 
          feature_fun=None, 
          rmax=None,
          normalize=True, 
          use_rmax=True, 
          use_renyi=True,
          max_offline_ite=100, 
          max_search_ite=30,
          verbose=True, 
          save_to=None,
          delta=0.2,
          shift=False,
          reuse=False,
          use_parabola=False):
    
    #Logging
    format_strs = []
    if verbose: format_strs.append('stdout')
    if save_to: format_strs.append('csv')
    logger.configure(dir=save_to, format_strs=format_strs)

    pol = pol_maker('pol')
    newpol = pol_maker('oldpol')
    newpol.set_params(pol.eval_params())
    
    #Learning iteration
    actor_params, rets, disc_rets, lens = [], [], [], []
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
            for ep in range(batch_size):
                frozen_pol = pol.freeze()
                theta = frozen_pol.resample()
                actor_params.append(theta)
                ret, disc_ret, ep_len = eval_trajectory(env, frozen_pol, gamma, task_horizon, feature_fun)
                rets.append(ret)
                disc_rets.append(disc_ret)
                lens.append(ep_len)
        logger.log('Performance: ', np.mean(rets))
        #if save_to: np.save(save_to + '/rets_' + str(it), rets)
            

        norm_disc_rets = np.array(disc_rets)
        if shift:
            norm_disc_rets = norm_disc_rets - np.mean(norm_disc_rets)
        rmax = np.max(abs(norm_disc_rets))
        
        #Offline optimization
        with timed('Optimizing offline'):
            rho, improvement = optimize_offline(pol, newpol, actor_params, norm_disc_rets,
                                                normalize=normalize,
                                                use_rmax=use_rmax,
                                                use_renyi=use_renyi,
                                                max_offline_ite=max_offline_ite,
                                                max_search_ite=max_search_ite,
                                                rmax=rmax,
                                                delta=delta,
                                                use_parabola=use_parabola)
            newpol.set_params(rho)
            #assert(improvement>=0.)
        
        logger.log('Recap of iteration %i' % it)
        eRenyi = np.exp(np.amax(newpol.eval_renyi(pol)))

        logger.record_tabular('eRenyi2', eRenyi)
        logger.record_tabular('AvgRet', np.mean(rets))
        logger.record_tabular('VanillaAvgRet', np.mean(rets))
        logger.record_tabular('VarRet', np.var(rets, ddof=1))
        logger.record_tabular('VarDiscRet', np.var(norm_disc_rets, ddof=1))
        logger.record_tabular('AvgDiscRet', np.mean(norm_disc_rets))
        logger.record_tabular('BatchSize', len(rets))
        logger.record_tabular('AvgEpLen', np.mean(lens))
        logger.dump_tabular()
        
        
        #Update behavioral
        pol.set_params(newpol.eval_params()) 
        
        if improvement>0 or not reuse:
            actor_params, rets, disc_rets, lens = [], [], [], []
        if len(rets)>=5*batch_size:
            actor_params = actor_params[batch_size:]
            rets = rets[batch_size:]
            disc_rets = disc_rets[batch_size:]
            lens = lens[batch_size:]
        
    