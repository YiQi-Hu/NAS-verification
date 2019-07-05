from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
subsample_ratio_range = np.arange(0.05, 1, 0.05)
epoch_range = range(0, 50)
# Make data.

X, Y = np.meshgrid(epoch_range,subsample_ratio_range )

Z = np.loadtxt("1res.txt",delimiter=" ")
f = open('trick.result', 'r')
k = 0
i = 0
j = 0
g = np.zeros([19, 49])
for line in f.readlines():
    k += 1
    if k % 2 == 0:
        g[i][j] = line.split(' = ')[1]
        j += 1
        if j == 49:
            j = 0
            i += 1
    else:
        continue
print(g)
Z = g
# Plot the surface.
ax.plot_surface(X,Y,Z)#, linewidth=0.2, antialiased=True)

plt.show()
