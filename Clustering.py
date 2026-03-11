# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 27, 48, 50],
    'Income': [20000, 22000, 50000, 52000, 48000, 55000, 21000, 23000, 51000, 53000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Select features
X = df[['Age', 'Income']]

# Create KMeans model
kmeans = KMeans(n_clusters=2)

# Fit model
kmeans.fit(X)

# Predict clusters
df['Cluster'] = kmeans.predict(X)

print(df)

# Plot clusters
plt.scatter(df['Age'], df['Income'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering")
plt.show()
