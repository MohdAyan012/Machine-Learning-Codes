import numpy as np
from sklearn.cluster import KMeans
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
print("Cluster labels:")
print(kmeans.labels_)
