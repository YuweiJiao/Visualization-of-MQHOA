import numpy as np
import math
import func
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time


def F(x):
    s = np.random.uniform(LB, UB, (Dim, k))
    for d in range(0, Dim):
        for m_i in range(0, k):
            s[d, m_i] = 418.9829 - x[d, m_i] * np.sin(np.sqrt(np.abs(x[d, m_i])))
    return s


"""
初始化
"""
Dim = 2
LB = -500
UB = 500
evo = Dim * 1e4
acc = 1e-6
sigma = UB - LB
k = 30
funcV = np.ones(k)
samplePos = np.random.uniform(LB, UB, (Dim, k))
optimalSolution = np.random.uniform(LB, UB, (Dim, k))
for i in range(0, k):
    funcV[i] = func.Schwefel(optimalSolution[:, i])
std = np.std(optimalSolution, axis=1)
# print (funcV)


while evo > 0:
    change_flag = 0
    sigma_c = np.array([sigma * 0.5] * Dim)
    cov = np.diag(sigma_c ** 2)
    for i in range(0, k):
        samplePos[:, i] = np.random.multivariate_normal(optimalSolution[:, i], cov)
        for j in range(Dim):
            if samplePos[j, i] < LB or samplePos[j, i] > UB:
                samplePos[j, i] = LB + (UB - LB) * np.random.random(1)
        sampleValue = func.Schwefel(samplePos[:, i])
        evo = evo - 1
        if sampleValue < funcV[i]:
            funcV[i] = sampleValue
            optimalSolution[:, i] = samplePos[:, i]
            change_flag = 1

    if change_flag == 0:
        s_matrix = F(optimalSolution)
        for d_i in range(0, Dim):
            badIndex = np.argmax(s_matrix[d_i, :])
            bestIndex = np.argmin(s_matrix[d_i, :])
            optimalSolution[d_i, badIndex] = optimalSolution[d_i, badIndex] + random.random() * (
                        optimalSolution[d_i, bestIndex] - optimalSolution[d_i, badIndex])
        for i in range(0, k):
            funcV[i] = func.Schwefel(optimalSolution[:, i])
        std = np.std(optimalSolution, axis=1)

    if std.max() < sigma:
        sigma = sigma / 2.0


print(funcV)
