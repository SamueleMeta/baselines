import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time


def plot_bound_profile(rho_grid, bound, mise, bonus,
                       point_x, point_y, iter, filename):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(rho_grid, bound, label='bound', color='red', linewidth='2')
    ax.plot(rho_grid, mise, label='mise', color='blue', linewidth='0.5')
    ax.plot(rho_grid, bonus, label='bonus', color='green', linewidth='0.5')
    ax.plot(point_x, point_y, 'o', color='orange')
    ax.legend(loc='upper right')
    # Save plot to given dir
    dir = './bound_profile/' + filename + '/'
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)


def plot3D_bound_profile(x, y, bound, rho_best, bound_best, iter, filename):

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.plot_surface(x, y, bound, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('gain')
    ax.set_ylabel('std')
    ax.set_zlabel('bound')
    ax.invert_yaxis()
    # y = np.exp(y)
    # rho_best[1] = np.exp(rho_best[1])
    ax.plot([rho_best[0]], [rho_best[1]], [bound_best],
            markerfacecolor='r', marker='o', markersize=5)
    # Save plot to given dir
    dir = './bound_profile/' + filename + '/'
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(fname)
    plt.close(fig)


def plot_ess(rho_grid, ess, iter, filename):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(rho_grid, ess, color='blue', linewidth='1')
    # Save plot to given dir
    dir = './bound_profile/' + filename + '/'
    siter = 'iter_{}'.format(iter)
    fname = dir + siter
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.savefig(fname)
    plt.close(fig)


def render(env, pi, horizon):
    """
    Shows a test episode on the screen
        env: environment
        pi: policy
        horizon: episode length
    """
    t = 0
    ob = env.reset()
    env.render()

    done = False
    while not done and t < horizon:
        ac, _ = pi.act(True, ob)
        ob, _, done, _ = env.step(ac)
        time.sleep(0.1)
        env.render()
        t += 1


##############################################################

def best_of_grid(policy, grid_size,
                 rho_init, old_rhos_list,
                 iters_so_far, mask_iters,
                 set_parameters, set_parameters_old,
                 delta_cst,
                 evaluate_behav, evaluate_bound,
                 evaluate_renyi, evaluate_roba,
                 filename):

    threeDplot = False

    # Compute MISE's denominator and Renyi bound
    den_mise = np.zeros(mask_iters.shape).astype(np.float32)
    # renyi_components_sum = 0
    for i in range(len(old_rhos_list)):
        set_parameters_old(old_rhos_list[i])
        behav = evaluate_behav()
        den_mise = den_mise + np.exp(behav)

    # Compute the log of MISE's denominator
    eps = 1e-24  # to avoid inf weights and nan bound
    den_mise = (den_mise + eps) / iters_so_far
    den_mise_log = np.log(den_mise) * mask_iters

    # Calculate the grid of parameters to evaluate
    gain_grid = np.linspace(-1, 1, grid_size)
    logstd_grid = np.linspace(-4, 0, grid_size)
    std_too = (len(rho_init) == 2)
    if std_too:
        threeDplot = True
        x, y = np.meshgrid(gain_grid, logstd_grid)
        X = x.reshape((np.prod(x.shape),))
        Y = y.reshape((np.prod(y.shape),))
        rho_grid = list(zip(X, Y))
    else:
        rho_grid = [[x] for x in gain_grid]
    # Evaluate the set of parameters and retain the best one
    bound = []
    mise = []
    bonus = []
    bound_best = 0
    renyi_bound_best = 0
    rho_best = rho_init

    for rho in rho_grid:
        set_parameters(rho)
        renyi_components_sum = 0
        for i in range(len(old_rhos_list)):
            set_parameters_old(old_rhos_list[i])
            renyi_component = evaluate_renyi()
            renyi_components_sum += 1 / renyi_component
        renyi_bound = 1 / renyi_components_sum
        bound_rho = evaluate_bound(den_mise_log, renyi_bound)
        bound.append(bound_rho)
        if not std_too:
            # Evaluate bounds' components for plotting
            mise_rho, bonus_rho = \
                evaluate_roba(den_mise_log, renyi_bound)
            mise.append(mise_rho)
            bonus.append(bonus_rho)

        if bound_rho > bound_best:
            bound_best = bound_rho
            rho_best = rho
            renyi_bound_best = renyi_bound

    # Plot the profile of the bound and its components
    if threeDplot:
        bound = np.array(bound).reshape((grid_size, grid_size))
        plot3D_bound_profile(x, y, bound, rho_best, bound_best,
                             iters_so_far, filename)
    else:
        plot_bound_profile(gain_grid, bound, mise, bonus, rho_best[0],
                           bound_best, iters_so_far, filename)

    # Calculate improvement
    set_parameters(rho_init)
    improvement = bound_best - evaluate_bound(den_mise_log, renyi_bound)

    return rho_best, improvement, den_mise_log, renyi_bound_best
