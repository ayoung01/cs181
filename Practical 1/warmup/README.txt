This program for the practical part has two files: K-Means.py, which runs the Lloyd's
algorithm on CIFAR-10 data set, and visualization.py, which visualizes the results from
K-Means.py using 32*32 .png images.

To use this program, first download CIFAR-10 data set from 
http://www.cs.toronto.edu/?kriz/cifar.html and then uncompress the file to get files files named 
data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5. After that, put the 
python file into the same directory as the five batch files and then run K-Means.py in terminal.
When K-Means.py finishes calculation, it will save its results into two files called 
cluster_centers.npy and responsibility_vectors.npy. Then run visualization.py in terminal, which
should automatically read in the data from cluster_centers.py and visualize it by generating K
different .png images and saving them in the same directory.