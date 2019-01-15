# coding: utf-8
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import baselines.common.tf_util as U
import time
from baselines import logger


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
        # Create GP regressor and fit it to the arms' returns
        gp = GaussianProcessRegressor()
        gp.fit(selected_rhos, selected_disc_rets)
        mu, sigma = gp.predict(rho_grid, return_std=True)


class GPUCB(object):
    def __init__(self, meshgrid, environment, beta=100.):
        '''
        meshgrid: Output from np.methgrid.
        e.g. np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)) for
        2D space with |x_i| < 1 constraint.
        environment: Environment class which is equipped with sample() method
        to return observed value.
        beta (optional): Hyper-parameter to tune the exploration-exploitation
        balance. If beta is large, it emphasizes the variance of the unexplored
        solution (i.e. larger curiosity)
        '''
        self.meshgrid = np.array(meshgrid)
        self.environment = environment
        self.beta = beta

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []

    def argmax_ucb(self):
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])
        gp = GaussianProcessRegressor()
        gp.fit(self.X, self.T)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        # print('self.X', self.X)
        # print('self.T', self.T)
        # print('self.mu', self.mu)
        print('self.X_grid', self.X_grid.shape)

    def sample(self, x):
        t = self.environment.sample(x)
        self.X.append(x)
        self.T.append(t)

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.mu.reshape(self.meshgrid[0].shape),
                          alpha=0.5, color='g')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.environment.sample(self.meshgrid),
                          alpha=0.5, color='b')
        ax.scatter([x[0] for x in self.X],
                   [x[1] for x in self.X],
                   self.T, c='r', marker='o', alpha=1.0)
        dir = './plots/'
        fname = dir + 'fig_%02d.png' % len(self.X)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(fname)


if __name__ == '__main__':
    class DummyEnvironment(object):
        def sample(self, x):
            return np.sin(x[0]) + np.cos(x[1])

    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    env = DummyEnvironment()
    agent = GPUCB(np.meshgrid(x, y), env)
    for i in range(10):
        print('----------Iter{}---------'.format(i+1))
        agent.learn()
        # agent.plot()
