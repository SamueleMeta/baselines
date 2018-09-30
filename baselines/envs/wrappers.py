import gym

class FixedHorizonWrapper(gym.Wrapper):

    def __init__(self, env, horizon=100):
        gym.Wrapper.__init__(self, env)
        self.horizon = horizon
        self.current_timestep = 0

    def reset(self):
        self.current_timestep = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_timestep += 1
        if self.current_timestep == self.horizon - 1:
            done = True
        return obs, reward, done, info
