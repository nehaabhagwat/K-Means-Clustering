import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Kmeans_Clustering:
	def __init__(self, k, verbose = False):
		self.k = k
		self.verbose = verbose
		# self.threshold = threshold
		self.centroids = []
		self.current_centroids = []
		self.clusters = np.empty((k,0)).tolist()
		self.y = []

	def distance(self, data1, data2, type = 'Euclidean'):
		if len(data1) != len(data2):
			if self.verbose == True:
				print "The length of the two data points is not equal. Please recheck the input data. Exiting...\n"
			exit()

		elif type == 'Euclidean':
			return np.linalg.norm(data1-data2)

		else:
			if self.verbose == True:
				print "Distance calculation for this distance type has not been implemented. Returning results for Euclidean distance...\n"
			return self.distance(data1, data2, type='Euclidean')

	def find_new_centroids(self):
		self.centroids = self.current_centroids
		self.current_centroids = []
		for cluster in self.clusters:
			self.current_centroids.append(np.average(cluster, axis=0))
			if self.verbose == True:
				print "Cluster centroid computed: " + str(self.current_centroids[-1])

	def assign_points_to_clusters(self, X_train):
		self.clusters = np.empty((self.k,0)).tolist()
		for point in X_train:
			min_dist, cluster_index = float("inf"), float("inf")
			for i in range(0, len(self.current_centroids)):
				# print point
				# print self.current_centroids[i]
				dist = self.distance(point, self.current_centroids[i])
				if dist < min_dist:
					min_dist = dist
					cluster_index = i
			if self.verbose == True:
				print "Assigning point " + str(point) + " to cluster index " + str(cluster_index)
			self.clusters[cluster_index].append(point)
			self.y.append(cluster_index)

		if self.verbose == True:
			print "Current clusters: ", self.clusters


	def implement_clustering(self, X_train, max_epoch = 100):
		X_train = np.array(X_train)

		distances = []
		seeds = random.sample(range(0,len(X_train)), self.k)

		for seed in seeds:
			self.current_centroids.append(np.array(X_train[seed]))

		iteration_count = 0 
		while(iteration_count<max_epoch):
			self.y = []
			self.assign_points_to_clusters(X_train)
			self.find_new_centroids()
			iteration_count += 1
		self.plot_graph()

		return self.clusters

	def plot_graph(self):
		plt.title("Clustered data for k = " + str(self.k))
		plt.xlabel("Duration")
		plt.ylabel("Weight")
		plt.scatter(X[:,0],X[:,1], marker = 'o', c = self.y, edgecolor = 'k')
		plt.show()

if __name__ == "__main__":
	X = np.array([[3.6,79],[1.8,54], [2.283,62], [3.333,74], [2.883,55], 
             [4.533,85], [1.950,51], [1.833,54], [4.700,88], [3.600,85], 
             [1.600,52], [4.350,85], [3.917,84],[4.200,78], [1.750,62],
            [1.800,51], [4.700,83], [2.167,52], [4.800,84],[1.750,47]
             ])

	clustering_object = Kmeans_Clustering(2)

	clusters = clustering_object.implement_clustering(X, 100)
	print "\n\nFor K = 2:\nClusters created are: " + str(clusters)

	clustering_object = Kmeans_Clustering(3)
	clusters = clustering_object.implement_clustering(X, 100)
	print "\n\nFor K = 3:\nClusters created are: " + str(clusters)




