#This file is a general plotting file for cda

import os 
import time 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cda_output(data, embedded_data, type, ksimgraph, error):
	final_directory = "dim_recution_"+str(round(time.time()))
	os.chdir("code_output")
	if(type == 0):
		os.chdir("cca")
	elif(type == 1):
		os.chdir("cda")
	os.makedirs(final_directory)
	os.chdir(final_directory)

	dat_array = np.array(data).transpose()
	embedded_dat_array = np.array(embedded_data).transpose()

	colors = np.zeros(shape=(len(data), 3))

	for i in range(len(data)):
		colors[i] = np.array([((data[i][0]*50)%255)/255.0, ((data[i][1]*50)%255)/255.0, 0])

	ax= plt.axes()

	ax.scatter(dat_array[0], dat_array[1], c=colors)
	plt.savefig('data_space_topdown.png')

	ax.clear()

	plot_2d_simgraph(data, embedded_data, ksimgraph)
	plot_3d_simgraph(data, ksimgraph)
	plot_3d_simgraph_no_edges(data, ksimgraph)

	ax= plt.axes()
	ax.plot(error)
	plt.savefig('error.png')
	

def plot_3d_simgraph_no_edges(data, adjmatrix):
	print("plotting 3d")
	
	colors = np.zeros(shape=(len(data), 3))

	for i in range(len(data)):
		colors[i] = np.array([((data[i][0]*50)%255)/255.0, ((data[i][1]*50)%255)/255.0, 0])
	
	fig = plt.figure()
	ax = Axes3D(fig)

	dat_array = np.array(data).transpose()
	ax.scatter(dat_array[0], dat_array[1], dat_array[2], c=colors)
	plt.savefig('data_space_no_edges.png')
	print("done plotting 3d")

def plot_3d_simgraph(data, adjmatrix):
	print("plotting 3d")
	edges = []
	for i in range(len(adjmatrix)):
		for j in range(len(adjmatrix)):
			if adjmatrix[i][j]<1.79769e308:
				edge = []
				edge.append(data[i])
				edge.append(data[j])
				edge.append(adjmatrix[i][j])
				edges.append(edge)
	
	colors = np.zeros(shape=(len(data), 3))

	for i in range(len(data)):
		colors[i] = np.array([((data[i][0]*50)%255)/255.0, ((data[i][1]*50)%255)/255.0, 0])
	
	fig = plt.figure()
	ax = Axes3D(fig)

	for edge in edges:
		xpoints = [edge[0][0], edge[1][0]]
		ypoints = [edge[0][1], edge[1][1]]
		zpoints = [edge[0][2], edge[1][2]]
		ax.plot(xpoints, ypoints, zpoints, c="gray", linewidth=.5)

	dat_array = np.array(data).transpose()
	ax.scatter(dat_array[0], dat_array[1], dat_array[2], c=colors)
	plt.savefig('data_space.png')
	print("done plotting 3d")

def plot_2d_simgraph(data, embedded_data, adjmatrix):
	print("plotting 2d")
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

	colors = np.zeros(shape=(len(data), 3))

	for i in range(len(data)):
		term1 = ((data[i][0]*50)%255)/255.0
		term2 = ((data[i][1]*50)%255)/255.0
		colors[i] = np.array([term1, term2, 0])

	fig = plt.figure()
	ax= plt.axes()

	for edge in edges:
		xpoints = [edge[0][0], edge[1][0]]
		ypoints = [edge[0][1], edge[1][1]]
		ax.plot(xpoints, ypoints, c="gray", linewidth=.5)

	dat_array= np.array(embedded_data).transpose();
	ax.scatter(dat_array[0], dat_array[1], c=colors)
	plt.savefig('embedding_space.png')
	print("done plotting 2d")