from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, LATENT_KEY
from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
from settings import SAMPLE_PATH_DB_KEY
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

edb = EmbeddingDatabaseManager()
psldb = ParametersShortLatentsDatabaseManager()

latents = edb.get_all_values_from_column(LATENT_KEY)
paths = edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)

num_objects = 2500
vector_length = 32000

num_representative_vectors = 4
latents_matrix = np.vstack(latents.values)

# Wykorzystanie algorytmu k-means
kmeans = KMeans(n_clusters=num_representative_vectors, random_state=42, max_iter=300, n_init=10)
kmeans.fit(latents_matrix)

# Centra klastrów
cluster_centers = kmeans.cluster_centers_

# Znalezienie wektorów najbliższych do centrów
representative_indices = []
for center in cluster_centers:
    distances = cdist([center], latents_matrix, metric='euclidean')
    closest_index = np.argmin(distances)
    representative_indices.append(closest_index)

# Wektory reprezentatywne
representative_vectors = latents.iloc[representative_indices]

# Analiza klastrów
cluster_labels = kmeans.labels_
cluster_sizes = pd.Series(cluster_labels).value_counts()

# Przykładowe wektory w każdym klastrze
clusters = {i: latents[cluster_labels == i].head(5) for i in range(num_representative_vectors)}

# Wyświetlenie wyników
print(f"Indeksy reprezentatywnych wektorów: {representative_indices}")
print(f"Ścieżki reprezentatywnych plików: {paths.iloc[representative_indices].values}")
print("\nRozmiary klastrów:")
print(cluster_sizes)

print("\nPrzykładowe wektory w klastrach:")
for cluster_id, examples in clusters.items():
    print(f"Klastr {cluster_id}:")
    print(examples)

# Wizualizacja z użyciem PCA
pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents_matrix)

# Tworzenie wykresu
plt.figure(figsize=(10, 8))
for cluster_id in range(num_representative_vectors):
    cluster_points = latents_2d[cluster_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Klastr {cluster_id}", alpha=0.6)

# Oznaczenie centrów klastrów
centers_2d = pca.transform(cluster_centers)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], color='black', marker='x', s=100, label='Centra klastrów')

plt.title("Wizualizacja klastrów przy użyciu PCA")
plt.xlabel("Pierwsza składowa PCA")
plt.ylabel("Druga składowa PCA")
plt.legend()
plt.grid()
plt.show()

score = silhouette_score(latents_matrix, cluster_labels)
print(f"Silhouette score: {score}")
print(f"Inercja k-means: {kmeans.inertia_}")
