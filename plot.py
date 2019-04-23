import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab as ml

ns = []
qs = []
fast_q = []
hitting_times = []
with open('mab/end_step.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        N, q, t = line.split()
        ns.append(int(N))
        qs.append(float(q))
        t = 3000 if int(t) > 3000 else int(t)
        hitting_times.append(int(t))
X = []
Y = []
with open('mab/results.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        N, q = line.split()
        X.append(int(N))
        Y.append(float(q))

xi = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
yi = np.linspace(0.25, 2.0, 8)
zi = ml.griddata(ns, qs, hitting_times, xi, yi, interp='linear')

plt.figure()
#zi = ml.griddata(X, Y, all_steps, xi, yi, interp='linear')
plt.contour(xi, yi, zi, levels=30, cmap=plt.cm.rainbow)
plt.colorbar(label='color')
plt.plot(X, Y, 'ks--')
#plt.ylim([0.0, 2.5])
plt.show()