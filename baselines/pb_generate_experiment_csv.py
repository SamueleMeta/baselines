script = ['pbpomis/run']

env = ['rllab.acrobot']

policy = ['linear']

delta = [0.6]

gamma = [1.]

njobs = [1]

seed = [749,	728,	524,	215,	455,	920,	635,	930,	402,	705,	938,	563,	925,	29,	173,	542,	899,	175,	152,	210]

capacity = [1]

num_episodes = [10]

max_iters = [5000]

iw_norm = ['all']

vars = dict(filter(lambda k: not k[0].startswith('__'), globals().items()))

keys = vars.keys()
values = vars.values()


from itertools import product
import csv

tuples = product(*values)

with open('pbpomis_14', mode='w') as file:
    employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(keys)
    for t in tuples:
        employee_writer.writerow(t)

print(len(list(tuples)))
