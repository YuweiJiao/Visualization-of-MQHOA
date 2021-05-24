import numpy as np
import math


def Sum_square(x):
    s = 0
    float(s)
    for i in range(len(x)):
        s += (i + 1) * x[i] ** 2
    return s


def Schwefel(matrix_x):
    s = 0
    x1 = matrix_x[0]
    x2 = matrix_x[1]
    #s=x1**2+x2**2
    s = 418.9829*2 - x1*np.sin(np.sqrt(np.abs(x1)))-x2*np.sin(np.sqrt(np.abs(x2)))
    return s
