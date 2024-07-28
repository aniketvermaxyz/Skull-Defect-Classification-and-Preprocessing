import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset (X: feature data)
X = X_pca # Features as a 2D array

# Define a range of number of clusters to try
min_clusters = 2
max_clusters = 10

# Perform K-means clustering for different number of clusters
inertias = []
silhouette_scores = []

for n_clusters in range(min_clusters, max_clusters + 1):
    # Create a K-means clustering object
    kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)

    # Perform clustering
    kmeans.fit(X)

    # Get inertia (sum of squared distances to the closest centroid)
    inertia = kmeans.inertia_
    inertias.append(inertia)

    # Get silhouette score
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(min_clusters, max_clusters + 1), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()


# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()



# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, n_init=30, random_state=42)
kmeans.fit(X)
labels__ = kmeans.labels_
labels__


# cluster_centers
from sklearn.decomposition import PCA

# Get cluster centers
cluster_centers = kmeans.cluster_centers_



# Plot the clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels__, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Result (PCA)')
plt.show()


