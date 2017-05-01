# # draw.py

import matplotlib.pyplot as plt
import csv

gen = []
for i in range(1, 10):
   gen.append(i)

data1 = [49421.0113083,
         49317.4063696,
         49181.0433395,
         49181.0433369,
         49134.3292262,
         49111.8690285,
         49128.8298889,
         49069.7810514,
         48956.7373088,
         ]

data2 = [49297.8708162,
         49173.9273485,
         49146.4561833,
         49119.2231553,
         49097.9996393,
         49055.9489897,
         49030.2677828,
         48955.3205046,
         48881.697453,
         ]

plt.plot(gen, data1, marker='o', color='r', label='Average Fitness')
plt.plot(gen, data2, marker='o', color='b', label='Best Fitness')
plt.xlabel("Generation")
plt.ylabel("Fit")
plt.title("Fitness level")
plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
# from csv import reader
#
# #Graph 1
# x1 = []
# y1 = []
#
# with open('avefit.csv', 'r') as f:
#     data1 = list(reader(f))
#
# for row in data1:
#     x1.append(row)
#
# plt.plot(x1)
# plt.xlabel('Generation')
# plt.ylabel('Average Fitness')
# plt.title('Fitness vs Generation Graph')
# plt.show()

