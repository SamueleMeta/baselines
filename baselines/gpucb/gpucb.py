# coding: utf-8
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import baselines.common.tf_util as U
import time
from baselines import logger
from plotting_tools import plot3D_bound_profile, plot_bound_profile


def eval_trajectory(env, pol, gamma, horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t < horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        # ob = np.reshape(ob, newshape=s.shape)
        ret += r
        disc_ret += gamma**t * r
        t += 1
        # Rescale episodic return in [0, 1] (Hp: r takes values in [0, 1])
        ret_rescaled = ret / horizon
        max_disc_ret = (1 - gamma**(horizon + 1)) / (1 - gamma)  # r =1,1,...
        disc_ret_rescaled = disc_ret / max_disc_ret

    return ret_rescaled, disc_ret_rescaled, t


def learn(make_env,
          make_policy,
          horizon,
          gamma=0.99,
          max_iters=1000,
          filename=None,
          grid_size=100,
          feature_fun=None,
          plot_bound=False):

    # Build the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Build the higher level policy
    pi = make_policy('pi', ob_space, ac_space)

    # Get all pi's learnable parameters
    all_var_list = pi.get_trainable_variables()
    var_list = \
        [v for v in all_var_list if v.name.split('/')[1].startswith('higher')]

    # TF functions
    set_parameters = U.SetFromFlat(var_list)
    get_parameters = U.GetFlat(var_list)

    # Generate the grid of parameters to evaluate
    gain_grid = np.linspace(-1, 1, grid_size)
    grid_size_std = int(grid_size)
    logstd_grid = np.linspace(-4, 0, grid_size_std)
    rho = get_parameters()
    std_too = (len(rho) == 2)
    if std_too:
        x, y = np.meshgrid(gain_grid, logstd_grid)
        X = x.reshape((np.prod(x.shape),))
        Y = y.reshape((np.prod(y.shape),))
        rho_grid = np.array(list(zip(X, Y)))
    else:
        rho_grid = np.array([[x] for x in gain_grid])

    # Learning loop
    beta = 100
    regret = 0
    mu = np.array([0. for _ in range(rho_grid.shape[0])])
    sigma = np.array([0.5 for _ in range(rho_grid.shape[0])])
    selected_rhos = []
    selected_disc_rets = []
    tstart = time.time()
    while True:
        iter += 1

        # Exit loop in the end
        if iter - 1 >= max_iters:
            print('Finished...')
            break

        # Learning iteration
        logger.log('********** Iteration %i ************' % iter)

        # Select the bound maximizing arm
        grid_idx = np.argmax(mu + sigma * np.sqrt(beta))
        rho_best = rho_grid[grid_idx]
        selected_rhos.append(rho_best)
        # Sample actor's parameters from chosen arm
        set_parameters(rho_best)
        _ = pi.resample()
        # Sample a trajectory with the newly parametrized actor
        _, disc_ret, _ = eval_trajectory(
            env, pi, gamma, horizon, feature_fun)
        selected_disc_rets.append(disc_ret)
        regret += (0.96512 - disc_ret)
        # Create GP regressor and fit it to the arms' returns
        gp = GaussianProcessRegressor()
        gp.fit(selected_rhos, selected_disc_rets)
        mu, sigma = gp.predict(rho_grid, return_std=True)

        # Store info about variables of interest
        if env.spec.id == 'LQG1D-v0':
            mu1_actor = pi.eval_actor_mean([[1]])[0][0]
            mu1_higher = pi.eval_higher_mean([[1]])[0]
            sigma = pi.eval_higher_std()[0]
            logger.record_tabular("LQGmu1_actor", mu1_actor)
            logger.record_tabular("LQGmu1_higher", mu1_higher)
            logger.record_tabular("LQGsigma_higher", sigma)
        logger.record_tabular("ReturnLastEpisode", disc_ret)
        logger.record_tabular("ReturnMean", sum(selected_disc_rets) / iter)
        logger.record_tabular("Regret", regret)
        logger.record_tabular("Regret/t", regret / iter)
        logger.record_tabular("Iteration", iter)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        # Plot the profile of the bound and its components
        if plot_bound:
            if std_too:
                ub = np.array(ub).reshape((grid_size_std, grid_size))
                plot3D_bound_profile(x, y, ub, rho_best, ub_best,
                                     iter, filename)
            else:
                plot_bound_profile(gain_grid, ub, average_ret, bonus, rho_best,
                                   ub_best, iter, filename)
        # Print all info in a table
        logger.dump_tabular()

    # Close environment in the end
    env.close()
