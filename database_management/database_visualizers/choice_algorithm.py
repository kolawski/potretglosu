def iterative_search(embedding_matrix, k=4, steps=10):
    """
    Interaktywny algorytm zawężania próbek na podstawie wyborów użytkownika.

    :param embedding_matrix: Macierz embeddingów (n próbek x wymiar embeddingu)
    :param k: Liczba próbek prezentowanych w każdym kroku
    :param steps: Maksymalna liczba kroków
    :return: Indeks najbardziej dopasowanej próbki
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    n_samples = embedding_matrix.shape[0]
    remaining_indices = list(range(n_samples))  # Zbiór wszystkich próbek

    for step in range(steps):
        # Wybierz k najbardziej rozproszonych próbek z pozostałych
        if len(remaining_indices) <= k:
            # Jeśli zostało mniej niż k próbek, zakończ iterację
            print(f"Krok {step + 1}: Pozostałe próbki {remaining_indices}")
            break

        # Oblicz dystanse tylko między pozostałymi próbkami
        remaining_embeddings = embedding_matrix[remaining_indices]
        distances = cdist(remaining_embeddings, remaining_embeddings, metric="euclidean")

        # Wybierz k najbardziej odległych próbek
        selected_indices = select_farthest_points_by_sum(distances, k=k)
        selected_samples = [remaining_indices[i] for i in selected_indices]

        # Wyświetl próbki użytkownikowi i poproś o wybór
        print(f"Krok {step + 1}: Wybrane próbki {selected_samples}")
        chosen_sample = int(input(f"Podaj indeks wybranej próbki z {selected_samples}: "))

        # Zawęź przestrzeń do najbliższych próbek w embeddingach
        chosen_embedding = embedding_matrix[chosen_sample]
        distances_to_chosen = cdist([chosen_embedding], embedding_matrix[remaining_indices], metric="euclidean")[0]
        remaining_indices = [remaining_indices[i] for i in np.argsort(distances_to_chosen)[:len(remaining_indices) // 2]]

    # Zwróć ostateczny wybór użytkownika
    return chosen_sample
