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
    parser.add_argument('--k', type=int, default=11, help='number of nearest neighbors for the k-means')
    parser.add_argument('--out_dim', type=int, default=2, help='dimensionality of the output')
    args = parser.parse_args()

    #read data from csv, convert to data matrix
    data = pandas.read_csv('toydata.csv').values

    #delete label row
    data = np.delete(data, 0, 0).astype(np.float)
    print(data)
    z = data[:,0]
    df = pandas.DataFrame(data)

    #calculate pairwise distances
    dist_matrix = pandas.DataFrame(distance_matrix(df.values, df.values)).values

    #find k closest neighbors for each data point
    D0 = np.zeros(dist_matrix.shape)

    for index in range(0, dist_matrix.shape[0]):
    	vector = dist_matrix[:,index]
    	k_smallest = np.argpartition(vector, args.k)[:args.k]
    	for k_index in k_smallest:
    		D0[k_index, index] = dist_matrix[k_index, index]

    #find distances in graph of k neighbors (dijkstra)
    shortest_paths = sparse.csgraph.shortest_path(D0,'D',False)

    #minimize error in distances in a lower dimensional space
    X = gradient_descent(shortest_paths, args.out_dim)

    rc = recall(dist_matrix,d_matrix(X),0.1)
    pr = precision(dist_matrix,d_matrix(X),0.1)
    
    print("precision:")
    print(pr)
    print("recall:")
    print(rc)

    plot_data(X, z)

def gradient_descent(D0, dim):
    data_length = D0.shape[0]

    #distance matrix in lower dimensional space
    D = np.random.rand(data_length, data_length)
    #data matrix in lower dimensional space
    X = np.random.rand(data_length, dim)

    iterations = 500
    for iteration in range(0, iterations):
        print("Iteration: " + str(iteration) + " / " + str(iterations))

        #update distance matrix
        df = pandas.DataFrame(X)
        D = pandas.DataFrame(distance_matrix(df.values, df.values)).values

        #calculate gradient
        d_d = (D0 - D) / D
        np.fill_diagonal(d_d,0)
        d_d_rowsum = d_d @ np.ones((data_length,dim))
        gradient = (d_d @ X - d_d_rowsum * X) * 2

        #normalize gradient to unit length
        magnitude = math.sqrt(np.sum(gradient**2))
        gradient = gradient / magnitude

        #update X
        X = X - (gradient * 0.5)

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

def d_matrix(X):

    df = pandas.DataFrame(X)
    D = pandas.DataFrame(distance_matrix(df.values, df.values)).values
    return D

def intersect(b1, b2):
    return [val for val in b1 if val in b2]

def precision(o, re, radius):

    original = np.copy(o)
    reduction = np.copy(re)

    #1 true, 0 false
    original[original > radius] = 0
    original[original > 0] = 1

    #2 true, -2 false
    reduction[reduction > radius] = -2
    reduction[reduction >= 0] = 2

    combined = original + reduction

    #if both conditions true 1, otherwise 0
    combined[combined < 0] = 0
    combined[combined % 2 == 0] = 0
    combined[combined > 0] = 1

    n = original.shape[0]

    pr = 0

    for i in range(0,n):
        c = np.argwhere(combined[i,:] == 1).tolist()
        o = np.argwhere(original[i,:] == 1).tolist()
        inter = len(intersect(c,o))
        pr += inter / len(o)

    return pr / n

def recall(o, re, radius):

    original = np.copy(o)
    reduction = np.copy(re)

    #1 true, 0 false
    original[original > radius] = 0
    original[original > 0] = 1

    #2 true, -2 false
    reduction[reduction > radius] = -2
    reduction[reduction >= 0] = 2

    combined = original + reduction

    #if both conditions true 1, otherwise 0
    combined[combined < 0] = 0
    combined[combined % 2 == 0] = 0
    combined[combined > 0] = 1

    n = original.shape[0]

    rc = 0

    for i in range(0,n):
        c = np.argwhere(combined[i,:] == 1).tolist()
        o = np.argwhere(reduction[i,:] == 2).tolist()
        inter = len(c)
        rc += inter / len(o)

    return rc / n

if __name__ == '__main__':
	main()