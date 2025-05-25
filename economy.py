# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv("Country-data.csv")

# Remove extra spaces from 'country' column
df['country'] = df['country'].str.strip()

# Drop the 'country' column from features
X = df.drop(columns=["country"])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (Optional) Convert scaled data into DataFrame (to preserve column names)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# (Optional) Print first 5 rows
# print(X_scaled_df.head())

# Create DBSCAN model
# eps: neighborhood radius (tune this parameter)
# min_samples: minimum number of points to form a cluster

dbscan = DBSCAN(eps=1.5, min_samples=5)

# Fit the model and get cluster labels
clusters = dbscan.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['cluster'] = clusters

# Reduce to 2 dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = df['cluster']
pca_df['country'] = df['country']

# Plot the PCA results with clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='tab10', s=100, legend='full')
plt.title("DBSCAN Clustering Results (Reduced to 2D with PCA)")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Print column names, cluster label counts, and data types
print(df.columns)
print(df['cluster'].value_counts())
print(df.dtypes)

# Calculate cluster-wise mean values
cluster_profiles = df.groupby('cluster').mean(numeric_only=True)
print(cluster_profiles)

# Print number of countries in each cluster
print(df['cluster'].value_counts())

# Print number of clusters and their profiles
print("Number of Clusters:")
print(df['cluster'].value_counts())
print("\nCluster Profiles:")
print(cluster_profiles)
input("\nCluster profiles have been generated. Press Enter to continue...")

# Print noise points (outliers)
print("\nNoise Points:")
print(df[df['cluster'] == -1][['country']])

# Save clustered data to CSV
df.to_csv("clustered_countries.csv", index=False)
