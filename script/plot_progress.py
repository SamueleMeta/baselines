import matplotlib.pyplot as plt
import csv

avg_return = []
bound = []
with open('progress.csv') as fp:
    reader = csv.DictReader(fp)
    for row in  reader:
        avg_return.append(float(row['EpRewMean']))
        bound.append(float(row['StudentTBound']))

for x in range(len(avg_return)):
    plt.axvline(x, color='y')
plt.plot(range(len(avg_return)),avg_return)
plt.plot(range(len(bound)), bound)

plt.show()

