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

    dist_matrix = pandas.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).values

    print(df)
    print(dist_matrix)

    print(dist_matrix)

    D0 = np.zeros(dist_matrix.shape)

    for index in range(0, dist_matrix.shape[0]):
    	vector = dist_matrix[:,index]
    	k_smallest = np.argpartition(vector, args.k)[:args.k]
    	print(args.k+1)
    	for k_index in k_smallest:
    		D0[k_index, index] = dist_matrix[k_index, index]

    print("D0:")
    print(D0)
    #a = np.column_stack((x, y))
    a = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    print(a)
    b = sparse.csgraph.shortest_path(D0,'D',False)
    print(b)

    dp = gradient_descent(b, args.out_dim, z)
    print(dp)

def gradient_descent(D0, dim, color):
	datapoints = D0.shape[0]
	print("datapoints " + str(datapoints))

	print("out dim " + str(dim))

	D = np.random.rand(datapoints, datapoints)
	X = np.random.rand(datapoints, dim)

	print("X: ")
	print(X)
	print("D: ")
	for iteration in range(0, 250):
		print("Iteration: ")
		print(iteration)
		#update distance matrix

		df = pandas.DataFrame(X, columns=['xcord', 'ycord'])
		D = pandas.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).values

		print("D: ")
		print(D)

		d_d = (D0 - D) / D
		np.fill_diagonal(d_d,0)

		d_d_rowsum = d_d @ np.ones((datapoints,dim))
		gradient = (d_d @ X - d_d_rowsum * X)* 2
		magnitude = math.sqrt(np.sum(gradient**2))
		gradient = gradient / magnitude
		print(d_d)
		print("gradient: ")
		print(gradient)
		#gradient2 = 
		X = X - (gradient * 0.8)

	print("Updated X: ")
	print(X)

	print("D")
	print(D)

	vector1 = X[:,0]
	vector2 = X[:,1]
	vector3 = vector2

	cm = plt.cm.get_cmap('RdYlBu')

	min_x = np.amin(vector1)-2
	max_x = np.amax(vector1)+2

	vector1
	sc = plt.scatter(vector1, vector2, c=color, vmin=0, vmax=1, s=2, cmap=cm)
	plt.colorbar(sc)
	plt.show()

	return datapoints

if __name__ == '__main__':
	main()