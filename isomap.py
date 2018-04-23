import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy import spatial
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import pandas
import math

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="toydata.csv", help='path of data file')
    parser.add_argument('--k', type=str, default=11, help='number of nearest neighbors for the k-means')
    parser.add_argument('--out_dim', type=str, default=2, help='dimensionality of the output')
    args = parser.parse_args()

    #read data from csv, convert to data matrix
    colnames = ['s', 'x1', 'x2']
    data = pandas.read_csv('toydata.csv', names=colnames)
    x0 = data.x1.tolist()[1:1000]
    y0 = data.x2.tolist()[1:1000]
    z0 = data.s.tolist()[1:1000]

    x = [float(i) for i in x0]
    y = [float(i) for i in y0]
    z = [float(i) for i in z0]

    data = np.column_stack((x,y))
    df = pandas.DataFrame(data, columns=['xcord', 'ycord'])

    #calculate pairwise distances
    dist_matrix = pandas.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).values

    #find k closest neighbors for each data point
    D0 = np.zeros(dist_matrix.shape)

    for index in range(0, dist_matrix.shape[0]):
    	vector = dist_matrix[:,index]
    	k_smallest = np.argpartition(vector, args.k)[:args.k]
    	print(args.k+1)
    	for k_index in k_smallest:
    		D0[k_index, index] = dist_matrix[k_index, index]

    #find distances in graph of k neighbors (dijkstra)
    shortest_paths = sparse.csgraph.shortest_path(D0,'D',False)
    print(shortest_paths)

    #minimize error in distances in a lower dimensional space
    X = gradient_descent(shortest_paths, args.out_dim)

    plot_data(X, z)

def gradient_descent(D0, dim):
	data_length = D0.shape[0]

	#distance matrix in lower dimensional space
	D = np.random.rand(data_length, data_length)
	#data matrix in lower dimensional space
	X = np.random.rand(data_length, dim)

	for iteration in range(0, 100):
		print("Iteration: " + str(iteration))

		#update distance matrix
		df = pandas.DataFrame(X, columns=['xcord', 'ycord'])
		D = pandas.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).values

		#calculate gradient
		d_d = (D0 - D) / D
		np.fill_diagonal(d_d,0)
		d_d_rowsum = d_d @ np.ones((data_length,dim))
		gradient = (d_d @ X - d_d_rowsum * X)* 2

		#normalize gradient to unit length
		magnitude = math.sqrt(np.sum(gradient**2))
		gradient = gradient / magnitude

		#update X
		X = X - (gradient * 0.2)

	print("Updated X: ")
	print(X)

	print("Pairwise distances in reduced dimension")
	print(D)

	return X

def plot_data(data_matrix, color):
    vector1 = data_matrix[:,0]
    vector2 = data_matrix[:,1]

    cm = plt.cm.get_cmap('RdYlBu')

    sc = plt.scatter(vector1, vector2, c=color, vmin=0, vmax=1, s=2, cmap=cm)
    plt.colorbar(sc)
    plt.show()


if __name__ == '__main__':
	main()