
from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plydata = PlyData.read('/Users/john/PycharmProjects/trail/python-plyfile-master/data/scene0001_00_vh_clean_2.labels.ply')
data=plydata.elements[0].data
data=list(zip(*data))
x=np.array(data[0])
y=np.array(data[1])
z=np.array(data[2])
r=list(data[3])
g=list(data[4])
b=list(data[5])
l=list(data[7])

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 散点图参数设置
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('3D')
ax.scatter3D(x, y, z, c='r', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
