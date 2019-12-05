script = ['pomis/run']

env = ['rllab.cartpole',
        'rllab.mountain-car',
        'rllab.inverted-pendulum',
        'rllab.acrobot',
        'rllab.inverted-double-pendulum']

policy = ['linear']

iw_method = ['is', 'pdis']

bound = ['max-d2-harmonic']

delta = [0.4, 0.6, 0.8, 0.9, 0.99, 0.999, 0.9999]

gamma = [1.]

njobs = [1]

seed = [10, 109, 904, 160, 570]

policy_init = ['zeros']

capacity = [1, 5, 10]

num_episodes = [10]

max_iters = [200]

vars = dict(filter(lambda k: not k[0].startswith('__'), globals().items()))

keys = vars.keys()
values = vars.values()


from itertools import product
import csv

tuples = product(*values)

with open('test.csv', mode='w') as file:
    employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(keys)
    for t in tuples:
        employee_writer.writerow(t)

print(len(list(tuples)))
