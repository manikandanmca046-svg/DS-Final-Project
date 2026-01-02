import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kmeans_models(df):

    # Select numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster labels
    df['Cluster'] = clusters

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        numeric_df.iloc[:, 0],
        numeric_df.iloc[:, 1],
        c=clusters
    )
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])
    plt.title("K-Means Clustering")
    plt.show()