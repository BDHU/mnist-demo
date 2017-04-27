# draw.py

import matplotlib.pyplot as plt
import csv

gen = []
for i in range(1, 21):
    gen.append(i)

data = []
with open('results.cvs', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)

plt.plot(gen, data, marker = 'o', color='r', label='Result')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness of Each Generation")
plt.show()