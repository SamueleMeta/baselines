script = ['pbpois/run']

env = ['rllab.cartpole']#,
"""
env=['rllab.mountain-car',
        'rllab.inverted-pendulum',
        'rllab.acrobot',
        'rllab.inverted-double-pendulum']
"""

policy = ['linear']

delta = [0.2]

gamma = [1.]

njobs = [1]

seed = [10, 109, 904, 160, 570, 662, 963, 100, 746, 236, 247, 689, 153, 947, 307, 42, 950, 315, 545, 178]

capacity = [1]

num_episodes = [10]

max_iters = [500]

iw_norm = ['rows']

vars = dict(filter(lambda k: not k[0].startswith('__'), globals().items()))

keys = vars.keys()
values = vars.values()


from itertools import product
import csv

tuples = product(*values)

with open('pbpomis_grid.csv', mode='w') as file:
    employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(keys)
    for t in tuples:
        employee_writer.writerow(t)

print(len(list(tuples)))
