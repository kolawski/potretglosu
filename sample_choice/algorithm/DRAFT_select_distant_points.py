import numpy as np
from scipy.spatial.distance import pdist, squareform
from settings import EMBEDDINGS_DB, SAMPLE_PATH_DB_KEY
from database_management.database_managers.embedding_database_manager \
    import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY

db_manager = EmbeddingDatabaseManager(db_path=EMBEDDINGS_DB)
embeddings = db_manager.get_all_values_from_column(EMBEDDING_KEY)
latents = db_manager.get_all_values_from_column(LATENT_KEY)
paths = db_manager.get_all_values_from_column(SAMPLE_PATH_DB_KEY)

#Funkcja do wyboru najbardziej odległych punktów
def select_maximally_distant_points(distance_matrix, k=4):
    """
    Wybiera k najbardziej odległych punktów, zaczynając od najbardziej skrajnego punktu.
    """
    n_points = distance_matrix.shape[0]
    initial_point = np.argmax(np.sum(distance_matrix, axis=0))  # Punkt o największej sumie odległości
    selected = [initial_point]

    for _ in range(k - 1):
        min_distances = np.min(distance_matrix[selected], axis=0)
        next_point = np.argmax(min_distances)  # Punkt najbardziej oddalony od wybranych
        selected.append(next_point)

    return selected


# Funkcja główna zwracająca 4 startowe próbki
def get_initial_samples(k=4):
    """
    Zwraca k najbardziej odległych próbek jako ścieżki, indeksy i embeddingi.
    """
    embedding_matrix, paths = load_data()

    # Oblicz macierz odległości
    distance_matrix = squareform(pdist(embedding_matrix, metric='euclidean'))

    # Wybierz najbardziej odległe punkty
    indices = select_maximally_distant_points(distance_matrix, k=k)

    # Zweryfikuj poprawność indeksów
    assert all(0 <= idx < len(paths) for idx in indices), "Indeksy wykraczają poza zakres listy ścieżek"

    return paths, indices, embedding_matrix


if __name__ == "__main__":
    paths, indices, embeddings = get_initial_samples()
    print("Wybrane ścieżki:")
    for idx in indices:
        print(f"Indeks: {idx}, Ścieżka: {paths[idx]}")
