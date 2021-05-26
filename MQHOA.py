import numpy as np
import math
import func
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

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

"""
画图
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.arange(LB, UB, 10)
Y = np.arange(LB, UB, 10)
X, Y = np.meshgrid(X, Y)
Z = 418.9829 * 2 - X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.2)
# plt.show()

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
        meanPos = np.mean(optimalSolution, axis=1)
        badIndex = np.argmax(funcV)
        std = np.std(optimalSolution, axis=1)
        optimalSolution[:, badIndex] = meanPos
        funcV[badIndex] = func.Schwefel(meanPos)
        evo = evo - 1
        std = np.std(optimalSolution, axis=1)
        sigma = sigma / 2.0

    if std.max() <= sigma:
        sigma = sigma / 2.0
    x = optimalSolution[0, :]
    y = optimalSolution[1, :]
    x = x.T
    y = y.T
    z = 418.9829 * 2 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))
    sca = ax.scatter(x, y, z, s=200, lw=0, c='red', alpha=0.5)
    print("Most fitted f: ", min(funcV))
    print("sigma: ", sigma)
    print("evo: ", 20000 - evo)
    if 'sca' in globals():
        sca.remove()
    sca = ax.scatter(x, y, z, s=40, lw=0, c='red', alpha=0.4)
    plt.pause(0.001)
    sca.remove()
plt.ioff();plt.show()

# print(funcV)
