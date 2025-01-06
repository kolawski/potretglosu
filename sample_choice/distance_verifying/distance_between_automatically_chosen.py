from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from scipy.spatial.distance import pdist, squareform
import numpy as np
from settings import EMBEDDINGS_DB, SAMPLE_PATH_DB_KEY

db_manager = EmbeddingDatabaseManager(db_path=EMBEDDINGS_DB)
unique_embeddings = db_manager.get_all_values_from_column()

# Konwertuj embeddingi na listę i macierz NumPy
data = unique_embeddings.tolist()
embedding_matrix = np.array(data)

# Oblicz macierz odległości
distance_matrix = squareform(pdist(embedding_matrix, metric='euclidean'))

def select_maximally_distant_points(distance_matrix, k=4):
    """
    Wybiera k najbardziej odległych punktów, zaczynając od najbardziej skrajnego punktu.
    """
    n_points = distance_matrix.shape[0]

    # Zacznij od punktu o największej sumie odległości do innych punktów
    initial_point = np.argmax(np.sum(distance_matrix, axis=0))
    selected = [initial_point]

    for _ in range(k - 1):
        # Oblicz minimalną odległość do już wybranych punktów dla każdego kandydata
        min_distances = np.min(distance_matrix[selected], axis=0)
        next_point = np.argmax(min_distances)  # Punkt najbardziej oddalony od wybranych
        selected.append(next_point)

    return selected


# Wybierz 4 najbardziej odległe punkty
indices = select_maximally_distant_points(distance_matrix, k=4)
selected_vectors = [data[i] for i in indices]
print("Wybrane indeksy:", indices)
print("Wybrane wektory:", selected_vectors)

# Wczytaj całą bazę danych jako DataFrame
df = db_manager._dd.compute()  # Konwersja Dask DataFrame na pandas DataFrame

# Pobierz ścieżki do plików na podstawie indeksów
file_paths = df.iloc[indices][SAMPLE_PATH_DB_KEY].tolist()

# Wyświetl ścieżki
print("Wybrane ścieżki do plików:")
for path in file_paths:
    print(path)

# SPRAWDZENIE

# Sprawdź odległości między wybranymi punktami
for i, idx1 in enumerate(indices):
    for j, idx2 in enumerate(indices):
        if i < j:  # Unikaj porównywania tych samych punktów
            dist = distance_matrix[idx1, idx2]
            print(f"Odległość między punktami {idx1} i {idx2}: {dist}")

# Wszystkie odległości w zbiorze
all_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

# Mediana i maksymalna odległość
median_distance = np.median(all_distances)
max_distance = np.max(all_distances)

print(f"Mediana odległości: {median_distance}")
print(f"Maksymalna odległość: {max_distance}")

# Sprawdź, jak odległości między wybranymi punktami wypadają na tle całego zbioru
selected_distances = [distance_matrix[i, j] for i in indices for j in indices if i < j]
print(f"Odległości między wybranymi punktami: {selected_distances}")
