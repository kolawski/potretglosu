import numpy as np
from scipy.spatial.distance import pdist, squareform
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from settings import EMBEDDINGS_DB, SAMPLE_PATH_DB_KEY
from database_management.database_managers.embedding_database_manager import EMBEDDING_KEY



db_manager = EmbeddingDatabaseManager(db_path=EMBEDDINGS_DB)


df = db_manager._dd.compute()
embedding_matrix = np.array(df[EMBEDDING_KEY].tolist())
paths = df[SAMPLE_PATH_DB_KEY].tolist()

# 1. Ścieżki do plików wybrane z wizualizacji t-SNE
selected_paths = [
    "/app/Resources/ready_audio_samples/common_voice_pl_37353781.wav",
    "/app/Resources/ready_audio_samples/common_voice_pl_21644938.wav",
    "/app/Resources/ready_audio_samples/common_voice_pl_20606419.wav",
    "/app/Resources/ready_audio_samples/common_voice_pl_20620396.wav",
]

# Mapowanie ścieżek na indeksy embeddingów
selected_indices = [paths.index(path) for path in selected_paths]
print("Wybrane indeksy:", selected_indices)

# 2. Obliczenie odległości między wybranymi punktami
selected_distances = [
    (i, j, np.linalg.norm(embedding_matrix[i] - embedding_matrix[j]))
    for idx, i in enumerate(selected_indices)
    for j in selected_indices[idx + 1 :]
]
print("Odległości między wybranymi punktami:")
for i, j, dist in selected_distances:
    print(f"Odległość między punktami {i} i {j}: {dist}")

# 3. Rozkład wszystkich możliwych odległości
all_distances = squareform(pdist(embedding_matrix, metric="euclidean"))  # Macierz odległości
all_distances_flat = all_distances[np.triu_indices_from(all_distances, k=1)]  # Wyciągnij górny trójkąt macierzy

# Statystyki całego zbioru
mean_distance = np.mean(all_distances_flat)
median_distance = np.median(all_distances_flat)
max_distance = np.max(all_distances_flat)
min_distance = np.min(all_distances_flat)

print("\nStatystyki odległości w całym zbiorze:")
print(f"Średnia odległość: {mean_distance}")
print(f"Mediana odległości: {median_distance}")
print(f"Maksymalna odległość: {max_distance}")
print(f"Minimalna odległość: {min_distance}")

# 4. Porównanie wybranych odległości z całym zbiorem
print("\nPorównanie odległości między wybranymi punktami a całym zbiorem:")
for i, j, dist in selected_distances:
    print(f"Odległość między {i} i {j}: {dist} (w stosunku do rozkładu: "
          f"{'poniżej' if dist < median_distance else 'powyżej'} mediany)")
