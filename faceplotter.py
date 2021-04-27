#This file was written to plot the output of cda on facespace


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
        choice = random.sample(range(0, len(embedded_array)), 1)[0]
        tooClose = False
        for point in indices:
            if(distance([embedded_array[point][0], embedded_array[point][1]], [embedded_array[choice][0], embedded_array[choice][1]]) < mindist):
                tooClose = True
        if not tooClose:
            indices.append(choice)
    return indices

def plot_dim_reduction(data, embedded_data, type, adjmatrix, error):
    final_directory = "dim_recution_"+str(round(time.time()))
    os.chdir("code_output")
    if(type == 0):
        os.chdir("cca")
    elif(type == 1):
        os.chdir("cda")
    os.makedirs(final_directory)
    os.chdir(final_directory)

    ndata= np.transpose(np.array([np.array(xi) for xi in data]).astype(np.float64))
    ndata_preimage = np.copy(ndata)
    ndata = np.transpose(ndata)
    embedded_array = np.array(embedded_data)

    randomlist = generatePseudoRandomList(embedded_array, 50, 1000000, 10)
    im = []
    for i in randomlist: #20
        test = ndata_preimage[:,i]
        test = np.transpose(np.resize(test, (64,64)))
        im.append(Image.fromarray((test*255).astype(np.uint8)))

    adjacency_data = []
    edges = []
    for i in range(len(adjmatrix)):
        for j in range(i+1, len(adjmatrix)):
            if adjmatrix[i][j]<1.79769e308:
                edge = []
                edge.append(embedded_data[i])
                edge.append(embedded_data[j])
                edge.append(adjmatrix[i][j])
                edges.append(edge)

    fig = plt.figure()
    ax= plt.axes()
    dat_array= np.transpose(embedded_array);
    ax.scatter(dat_array[0], dat_array[1])

    for edge in edges:
        xpoints = [edge[0][0], edge[1][0]]
        ypoints = [edge[0][1], edge[1][1]]
        ax.plot(xpoints, ypoints, c="gray", linewidth=.1)

    for imageindex in range(len(im)):
        imagebox = OffsetImage(im[imageindex], zoom=.2, cmap="Greys", norm=plt.Normalize(0,255))
        ab = AnnotationBbox(imagebox,(embedded_array[randomlist[imageindex]][0], embedded_array[randomlist[imageindex]][1]), xycoords='data', pad=0.0)
        ax.add_artist(ab)
    plt.savefig('cda_face_space.png')
