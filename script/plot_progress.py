import matplotlib.pyplot as plt
import csv

field = 'EpRewMean' 
avg_return = []
with open('progress.csv') as fp:
    reader = csv.DictReader(fp)
    for row in  reader:
        avg_return.append(float(row[field]))

plt.plot(range(len(avg_return)),avg_return)
plt.show()
