# draw.py

#import matplotlib.pyplot as plt
#import csv
#
#gen = []
#for i in range(1, 10):
#    gen.append(i)
#
#data1 = []
#with open('avefit.cvs', 'rb') as f:
#    reader = csv.reader(f)
#    data1.append(reader)
#data2 = []
#with open('bestfit.cvs', 'rb') as f:
#    reader = csv.reader(f)
#    data2.append(reader)
#plt.plot(gen, data1, marker = 'o', color='r', label='Average Fitness')
#plt.plot(gen, data2, marker = 'o', color='b', label='Best Fitness')
#plt.xlabel("Generation")
#plt.ylabel("Fit")
#plt.title("Fitness level")
#plt.show()
import numpy as np
import matplotlib.pyplot as plt
from csv import reader

#Graph 1
x1 = []
y1 = []

with open('avefit.cvs', 'r') as f:
    data1 = list(reader(f))
    
for row in data1:
    x1.append(row[0])
    y1.append(row[1])
    
plt.plot(x1,y1)
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('Fitness vs Generation Graph')
plt.show()

