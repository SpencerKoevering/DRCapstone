import os 
import time 
from sklearn import manifold
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateDeterministicSwissRoll(num_points):
    data= []
    for i in range(num_points):
        phi = (.1*i+1);
        for j in range(3):
            temp = []
            temp.append(10*phi*(math.cos(phi)))
            temp.append(10*phi*(math.sin(phi)))
            temp.append(j*8)
            data.append(temp)
    return data

final_directory = "dim_recution_"+str(round(time.time()))
os.chdir("code_output")
os.chdir("isomap")
os.makedirs(final_directory)
os.chdir(final_directory)


data = generateDeterministicSwissRoll(80)
ndata=np.array([np.array(xi) for xi in data])
ndata_preimage = np.copy(ndata)
#defaults to 5 neighbors, in kclosest neighbors graph
embedded_array = manifold.Isomap(n_neighbors=15, n_components=2).fit_transform(ndata)
# embedded_array = manifold.Isomap(n_neighbors=5, n_components=2).fit_transform(ndata) //200pt example

colors = np.zeros(shape=(len(data), 3))
for i in range(len(data)):
	colors[i] = np.array([((data[i][0]*50)%255)/255.0, ((data[i][1]*50)%255)/255.0, 0])

fig = plt.figure()
ax= plt.axes()
dat_array= embedded_array.transpose();
ax.scatter(dat_array[0], dat_array[1], c=colors)
plt.savefig('isomap_embedding_space.png')

fig = plt.figure()
ax = Axes3D(fig)

dat_array = ndata_preimage.transpose()
ax.scatter(dat_array[0], dat_array[1], dat_array[2], c=colors)
plt.savefig('isomap_data_space.png')