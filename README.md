# DRCapstone
Some dimensionality reduction code I wrote

Isomap_face_data.mat is some face data that is used as an example.


cca.cpp and cda.cpp are an implementation of curvilinear component/distance analysis that I wrote. These are the key files here


cca.py was an early attempt at cca in python, it is slow and I cannot vouch that it is bug-free


all of the \*plotter.py files are plotting functions that I wrote for various runs of the algorithms, you may find them helpful. Especially if you check how I plot the graph that cda outputs.


the .sh files are quick compilation commands for the .cpp files. They assume that you have installed opencv2, however I have disabled that library in the code, so the import is not necessary. It would probably be best for you to dice up the compilation scripts to do what you need.


the faceMatrix.csv file is the facedata mat file again, but in csv.


isomap and faceIsomap.py are the code that I used to generate my Isomap examples. As Isomap is implemented by scikit-learn its here for documentation rather than for use.


the vector_operations file contains some key vector arithmetic that both cca.cpp and cda.cpp use.
