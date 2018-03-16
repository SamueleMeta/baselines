import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.stats as sts

delta = .05

avg_return = []
bound = []
var = []
with open('../results/trpo/cheetah/progress.csv') as fp:
    reader = csv.DictReader(fp)
    for row in  reader:
        avg_return.append(float(row['EpRewMean']))
        J_hat = float(row['J_hat'])
        var_J = float(row['Var_J'])
        batch_size = int(float(row['EpLenMean']))
        b = J_hat - sts.t.ppf(1-delta, batch_size -1) * \
                     np.sqrt(var_J/batch_size)
        bound.append(min(b, 4000))

"""
for x in range(len(avg_return)):
    plt.axvline(x, color='y')
"""
plt.plot(range(len(avg_return)),avg_return)
plt.plot(range(len(bound)), bound)
plt.show()

