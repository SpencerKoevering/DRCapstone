#A general plotting function for CCA output. Must be 3d to 2d reduction.

import os 
import time 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dim_reduction(data, embedded_data, type):

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
	
	fig = plt.figure()
	ax = Axes3D(fig)


	ax.scatter(dat_array[0], dat_array[1], dat_array[2], c=colors)
	plt.savefig('data_space.png')

	ax.clear()
	ax= plt.axes()

	ax.scatter(dat_array[0], dat_array[1], c=colors)
	plt.savefig('data_space_topdown.png')

	ax.clear()
	
	ax.scatter(embedded_dat_array[0], embedded_dat_array[1], c=colors)
	plt.savefig('embedding_space.png')
