#This was the code run to get the isomap embedding of facespace

import os 
import time 
from sklearn import manifold
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from sklearn.metrics.pairwise import euclidean_distances
import random
import csv
with open('faceMatrix.csv') as csvfile:
	csvreader = csv.reader(csvfile)
	rows = []
	for row in csvreader:
		rows.append(row)

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


def readcsv():
	return rows

def distance(x,y):
      x=np.array(x)
      y=np.array(y)
      p=np.sum((x-y)**2)
      d=np.sqrt(p)
      return d

def generatePseudoRandomList(embedded_array, n, maxiters, mindist):
    indices = []
    iters = 0
    while(len(indices) < n and iters < maxiters):
        choice = random.sample(range(0, len(embedded_array)), 1)
        tooClose = False
        for point in indices:
            if(distance([embedded_array[point][0][0], embedded_array[point][0][1]], [embedded_array[choice][0][0], embedded_array[choice][0][1]]) < mindist):
                tooClose = True
        if not tooClose:
            indices.append(choice)
    return indices

final_directory = "dim_recution(imagetest)_"+str(round(time.time()))
os.chdir("code_output")
os.chdir("isomap")
os.makedirs(final_directory)
os.chdir(final_directory)

data = readcsv()
ndata=np.array([np.array(xi) for xi in data]).astype(np.float64)
ndata_preimage = np.copy(ndata)
ndata = np.transpose(ndata)


#defaults to 5 neighbors, in kclosest neighbors graph
embedded_array = manifold.Isomap(n_neighbors=5, n_components=2).fit_transform(ndata)

im = []
randomlist = generatePseudoRandomList(embedded_array, 50, 1000000, 10)
for i in randomlist: #20
    test = ndata_preimage[:,i]
    test = np.transpose(np.resize(test, (64,64)))
    im.append(Image.fromarray((test*255).astype(np.uint8)))


fig = plt.figure()
ax= plt.axes()
dat_array= np.transpose(embedded_array);
ax.scatter(dat_array[0], dat_array[1])

for imageindex in range(len(im)):
    imagebox = OffsetImage(im[imageindex], zoom=.2, cmap="Greys", norm=plt.Normalize(0,255))
    ab = AnnotationBbox(imagebox,(embedded_array[randomlist[imageindex]][0][0], embedded_array[randomlist[imageindex]][0][1]), xycoords='data', pad=0.0)
    ax.add_artist(ab)
plt.savefig('isomap_embedding_space.png')