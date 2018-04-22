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
    parser.add_argument('--k', type=str, default=10, help='number of nearest neighbors for the k-means')
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

    dist_matrix = pandas.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)

    print(df)
    print(dist_matrix)
    #a = np.column_stack((x, y))
    a = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    print(a)
    b = sparse.csgraph.shortest_path(dist_matrix,'D',False)
    print(b)

    dp = gradient_descent(b, args.out_dim, z)
    print(dp)

def gradient_descent(distance_matrix, dim, color):
	datapoints = distance_matrix.shape[0]
	print("datapoints " + str(datapoints))

	print("out dim " + str(dim))

	D = np.random.rand(datapoints, datapoints)
	X = np.random.rand(datapoints, dim)

	print("X: ")
	print(X)
	print("D: ")
	for iteration in range(0, 15):
		print("Iteration: ")
		print(iteration)
		#update distance matrix
		for i in range(0, datapoints):
			vec_i = X[i,:]
			for j in range(0, datapoints):
				vec_j = X[j,:]
				#difference = vec_i - vec_j
				D[i,j] = math.sqrt((X[i,0]-X[j,0])*(X[i,0]-X[j,0])+(X[i,1]-X[j,1])*(X[i,1]-X[j,1]))
				#D[i,j] = np.linalg.norm(difference)

		print("D: ")
		print(D)
		gradient = np.zeros((datapoints, dim))

		i = 0

		for point1 in range(0, datapoints):
			for point2 in range(0, datapoints):
				if point1 != point2:
					for dimension in range(0, dim):
						i += 1
						if i % 10000 == 0:
							print("i : " + str(i))
						addition = 2 * (distance_matrix[point1, point2] - D[point1,point2])
						addition = - addition * (X[point1,dimension] - X[point2,dimension]) / D[point1,point2]
						gradient[point1,dimension] = gradient[point1,dimension] + addition

		magnitude = math.sqrt(np.sum(gradient**2))
		gradient = gradient / magnitude

		d_d = (distance_matrix - D) / D
		np.fill_diagonal(d_d,0)

		d_d_rowsum = d_d @ np.ones((datapoints,dim))
		print(d_d_rowsum)
		gradient2 = (d_d @ X - d_d_rowsum * X)* 2
		magnitude2 = math.sqrt(np.sum(gradient2**2))
		gradient2 = gradient2 / magnitude2
		print(d_d)
		print("gradient: ")
		print(gradient)
		print("gradient2: ")
		print(gradient2)
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