from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

a = np.linspace(0, 5, 100)
r = np.linspace(1, 5, 100)
y = np.linspace(-1, 1, 100)

plt.subplot(121)
for r in np.arange(1, 2, 0.1):
    res = [np.exp((1+1*np.abs(y[i]))**r-1) for i in range(len(y))]
    plt.plot(res)
plt.subplot(122)
for a in np.arange(0, 2, 0.2):
    res = [np.exp((1+a*np.abs(y[i]))**1.6-1) for i in range(len(y))]
    plt.plot(res)
# 1.6, 1.6
plt.show()
