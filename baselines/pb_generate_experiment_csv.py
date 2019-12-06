script = ['pbpomis/run']

env = ['rllab.cartpole']#,
        #'rllab.mountain-car',
        #'rllab.inverted-pendulum',
        #'rllab.acrobot',
        #'rllab.inverted-double-pendulum']

policy = ['linear']

delta = [0.05, 0.1, 0.2, 0.3, 0.4]

gamma = [1.]

njobs = [1]

seed = [10, 109, 904, 160, 570]

capacity = [1, 10]

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
