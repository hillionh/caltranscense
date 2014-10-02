# coding utf-8

import random
import matplotlib.pyplot as plt
import numpy as np

from GMM import EM_algorithm


s = []
for i in range(5):
    s += [[]]
    for k in range(10):
        s[-1] += [np.matrix([random.gauss(5, 3)]).transpose()]

model = EM_algorithm(S = s, max_iter = 50)
print(model.parameters['means'])
print(model.parameters['variances'])
