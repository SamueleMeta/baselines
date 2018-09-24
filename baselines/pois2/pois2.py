'''
    GPU-friendly implementation of the POIS algorithm, for now only in the
    Control-based setting.
'''

def learn(env,
          make_policy):

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = make_policy('pi', ob_space, ac_space)

    # Stay there
    while True:
        pass
