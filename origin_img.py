import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('all_data.csv')[:100]
print(data.head())
fig = plt.figure()
ax = Axes3D(fig)
x = data['Unnamed: 0']
y = data['0']
z = data['1']
k = data['2']
o = data['3']
h1 = np.cos(np.sqrt(x**2+y**2))
h2 = np.cos(np.sqrt(x**2+z**2))
h3 = np.cos(np.sqrt(x**2+k**2))
h4 = np.cos(np.sqrt(x**2+o**2))

ax.scatter(x, y, h1)
ax.scatter(x, z, h2)
ax.scatter(x, k, h3)
ax.scatter(x, o , h4)

plt.show()
# plt.show()