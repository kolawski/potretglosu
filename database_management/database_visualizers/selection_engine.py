import numpy as np
from scipy.spatial.distance import cdist

class SelectionEngine:
    def __init__(self, embedding_matrix, paths, k=4):
        """
        Inicjalizuje silnik wyboru próbki.

        :param embedding_matrix: Macierz embeddingów
        :param paths: Ścieżki do plików audio
        :param k: Liczba próbek w każdej iteracji
        """
        self.embedding_matrix = embedding_matrix
        self.paths = paths
        self.k = k
        self.remaining_indices = list(range(len(embedding_matrix)))  # Wszystkie dostępne indeksy
        self.current_iteration = 0
        self.chosen_indices = []

    def get_next_samples(self):
        """
        Zwraca kolejne próbki do odsłuchu.

        :return: Lista ścieżek do wybranych próbek
        """
        if len(self.remaining_indices) <= self.k:
            return [self.paths[i] for i in self.remaining_indices]

        # Wybierz losowo k próbek
        selected_indices = np.random.choice(self.remaining_indices, self.k, replace=False)
        self.current_indices = selected_indices
        return [self.paths[i] for i in selected_indices]

    def update_selection(self, chosen_index):
        """
        Aktualizuje stan na podstawie wyboru użytkownika.

        :param chosen_index: Indeks wybranej próbki w aktualnym zestawie
        """
        chosen_embedding = self.embedding_matrix[self.current_indices[chosen_index]]
        remaining_embeddings = np.array([self.embedding_matrix[i] for i in self.remaining_indices])

        # Oblicz odległości do wybranej próbki
        distances = cdist([chosen_embedding], remaining_embeddings, metric="euclidean")[0]

        # Zachowaj tylko najbliższe próbki
        half_size = len(self.remaining_indices) // 2
        nearest_indices = np.argsort(distances)[:half_size]
        self.remaining_indices = [self.remaining_indices[i] for i in nearest_indices]

        self.chosen_indices.append(self.current_indices[chosen_index])

    def get_final_selection(self):
        """
        Zwraca ostateczną najlepszą próbkę.

        :return: Ścieżka do najlepszej próbki
        """
        return self.paths[self.remaining_indices[0]]
