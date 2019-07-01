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
# Plot the surface.
ax.plot_surface(X,Y,Z)#, linewidth=0.2, antialiased=True)

plt.show()
