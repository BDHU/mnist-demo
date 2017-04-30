# draw.py

import matplotlib.pyplot as plt
import csv

gen = []
for i in range(1, 10):
    gen.append(i)

data1 = []
with open('avefit.cvs', 'rb') as f:
    reader = csv.reader(f)
    data1 = list(reader)
data2 = []
with open('bestfit.cvs', 'rb') as f:
    reader = csv.reader(f)
    data2 = list(reader)
plt.plot(gen, data1, marker = 'o', color='r', label='Ave')
plt.plot(gen, data2, marker = 'o', color='b', label='Best')
plt.xlabel("Generation")
plt.ylabel("Fit")
plt.title("Fitness level")
plt.show()