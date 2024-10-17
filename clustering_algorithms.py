# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!
#%%
import numpy as np
from sklearn.utils import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

# class CustomKMeans:
#     def __init__(self, n_clusters=8, max_iter=300, random_state=None):
#         """
#         Creates an instance of CustomKMeans.
#         :param n_clusters: Amount of target clusters (=k).
#         :param max_iter: Maximum amount of iterations before the fitting stops (optional).
#         :param random_state: Initialization for randomizer (optional).
#         """
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.random_state = random_state
#         self.cluster_centers_ = None
#         self.labels_ = None

#     def fit(self, X: np.ndarray, y=None):
#         """
#         This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
#         will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
#         this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
#         the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
#         and/or encapsulate the necessary mechanisms in additional functions.
#         :param X: Array that contains the input feature vectors
#         :param y: Unused
#         :return: Returns the clustering object itself.
#         """
#         # Input validation:
#         X = check_array(X, accept_sparse='csr')

#         # Calculation of cluster centers:
#         self.cluster_centers_ = None  # TODO: Implement your solution here!

#         # Determination of labels:
#         self.labels_ = None  # TODO: Implement your solution here!

#         return self

#     def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
#         """
#         Calls fit() and immediately returns the labels. See fit() for parameter information.
#         """
#         self.fit(X)
#         return self.labels_
#%%
class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        cluster_centers = X[random_idx[:self.n_clusters]]
        return cluster_centers

    def assign_labels(self, X, cluster_centers):
        distances = euclidean_distances(X, cluster_centers)
        return np.argmin(distances, axis=1)

    def update_centers(self, X, labels):
        cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return cluster_centers

    def fit(self, X: np.ndarray, y=None):
        X = np.array(X)  # Ensure X is a numpy array for manipulation
        self.cluster_centers_ = self.initialize_centers(X)
        for _ in range(self.max_iter):
            labels = self.assign_labels(X, self.cluster_centers_)
            new_centers = self.update_centers(X, labels)
            if np.all(new_centers == self.cluster_centers_):
                break
            self.cluster_centers_ = new_centers
        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X)
        return self.labels_

# class CustomDBSCAN:
#     def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
#         """
#         Creates an instance of CustomDBSCAN.
#         :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
#         :param eps: Short for epsilon. Radius of considered circle around a possible core object.
#         :param metric: Used metric for measuring distances (optional).
#         """
#         self.eps = eps
#         self.min_samples = min_samples
#         self.metric = metric
#         self.labels_ = None

#     def fit(self, X: np.ndarray, y=None):
#         """
#         This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
#         will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
#         this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
#         long as it does this, you may change the content of this method completely and/or encapsulate the necessary
#         mechanisms in additional functions.
#         :param X: Array that contains the input feature vectors
#         :param y: Unused
#         :return: Returns the clustering object itself.
#         """
#         # Input validation:
#         X = check_array(X, accept_sparse='csr')

#         # Determination of labels:
#         self.labels_ = None  # TODO: Implement your solution here!

#         return self

#     def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
#         """
#         Calls fit() and immediately returns the labels. See fit() for parameter information.
#         """
#         self.fit(X)
#         return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        X = np.array(X)  # Ensure X is a numpy array for manipulation
        neighbors_model = NearestNeighbors(radius=self.eps, metric=self.metric)
        neighbors_model.fit(X)
        # Find the neighbors for each point
        neighbors = neighbors_model.radius_neighbors(X, return_distance=False)
        
        # Label initialization (-1 for unclassified)
        labels = np.full(shape=X.shape[0], fill_value=-1)
        cluster_label = 0
        
        # Iterate over each point
        for point_idx in range(X.shape[0]):
            # If the point is already classified, skip it
            if labels[point_idx] != -1:
                continue
            # If the point is a core point
            if len(neighbors[point_idx]) >= self.min_samples:
                # Start a new cluster
                labels[point_idx] = cluster_label
                # Prepare a list of points to be searched for neighbor points
                points_to_search = list(neighbors[point_idx])
                while points_to_search:
                    current_point = points_to_search.pop()
                    if labels[current_point] == -1:
                        labels[current_point] = cluster_label
                        if len(neighbors[current_point]) >= self.min_samples:
                            points_to_search.extend(neighbors[current_point])
                # Move on to the next cluster
                cluster_label += 1
        
        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X)
        return self.labels_

# %%
