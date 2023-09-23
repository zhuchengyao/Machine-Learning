import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# ax = fig.gca(projection='3d')

ax = fig.add_axes(Axes3D(fig))

plt.show()