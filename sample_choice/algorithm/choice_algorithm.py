from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
#from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
from settings import SAMPLE_PATH_DB_KEY
import numpy as np
#import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from utils.xtts_handler import XTTSHandler
from pathlib import Path
from sample_choice.algorithm.gender_list import man_indexes, women_indexes

edb = EmbeddingDatabaseManager()
xtts_handler = XTTSHandler()
#psldb = ParametersShortLatentsDatabaseManager()

embeddings = edb.get_all_values_from_column(EMBEDDING_KEY)
latents = edb.get_all_values_from_column(LATENT_KEY)
paths = edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)

print(embeddings.iloc[0].size)

# Przekształcenie latents na macierz NumPy
latents_matrix = np.vstack(latents.values)


# PCA na pełnym zbiorze
pca = PCA(n_components=3)
latents_3d = pca.fit_transform(latents_matrix)

def get_centroids(iteration, min_b, max_b):
    mid_x = (min_b[0] + max_b[0]) / 2
    mid_y = (min_b[1] + max_b[1]) / 2
    mid_z = (min_b[2] + max_b[2]) / 2

    if iteration == 0:
        # Iteracja 0 (krok 1): 8 punktów
        centroids = [
            [min_b[0], min_b[1], min_b[2]],
            [min_b[0], min_b[1], max_b[2]],
            [min_b[0], max_b[1], min_b[2]],
            [min_b[0], max_b[1], max_b[2]],
            [max_b[0], min_b[1], min_b[2]],
            [max_b[0], min_b[1], max_b[2]],
            [max_b[0], max_b[1], min_b[2]],
            [max_b[0], max_b[1], max_b[2]],
        ]
    elif iteration == 1:
        # Iteracja 1 (krok 2): 6 punktów
        centroids = [
            [min_b[0], min_b[1], mid_z],
            [min_b[0], max_b[1], mid_z],
            [mid_x, min_b[1], mid_z],
            [mid_x, max_b[1], mid_z],
            [max_b[0], min_b[1], mid_z],
            [max_b[0], max_b[1], mid_z],
        ]
    elif 2 <= iteration <= 6:
        # Iteracje 2–6 (kroki 3–7): 4 punkty
        centroids = [
            [mid_x, min_b[1], min_b[2]],
            [mid_x, max_b[1], min_b[2]],
            [mid_x, min_b[1], max_b[2]],
            [mid_x, max_b[1], max_b[2]],
        ]
    elif 7 <= iteration <= 8:
        # Iteracje 7–8 (kroki 8–9): 3 punkty
        centroids = [
            [min_b[0], mid_y, mid_z],
            [mid_x, max_b[1], mid_z],
            [max_b[0], mid_y, mid_z],
        ]
    else:
        # Iteracja 9 (krok 10): 2 punkty
        centroids = [
            [mid_x, min_b[1], mid_z],
            [mid_x, max_b[1], mid_z],
        ]
    return centroids

def find_nearest_indices(remaining_indices, centroids, pca, latents_matrix):
    remaining_latents_3d = pca.transform(latents_matrix[remaining_indices])
    selected_indices = []
    for centroid in centroids:
        distances = cdist(remaining_latents_3d, [centroid])
        nearest_index = np.argmin(distances)
        selected_indices.append(nearest_index)
    return selected_indices

min_bounds = np.min(latents_3d, axis=0)
max_bounds = np.max(latents_3d, axis=0)

# Nowe zapytanie na starcie
group_choice = input("m czy k? ")

if group_choice == 'm':
    remaining_indices = man_indexes.copy()
    ratio = 0.5
elif group_choice == 'k':
    remaining_indices = women_indexes.copy()
    ratio = 0.65
else:
    print("Niepoprawny wybór. Wybierz 'm' lub 'k'.")
    exit(1)

# Wykonujemy PCA tylko na wybranej grupie
pca = PCA(n_components=3)
pca.fit(latents_matrix[remaining_indices])
group_latents_3d = pca.transform(latents_matrix[remaining_indices])
min_bounds = np.min(group_latents_3d, axis=0)
max_bounds = np.max(group_latents_3d, axis=0)

iteration = 0
centroids = get_centroids(iteration, min_bounds, max_bounds)
selected_indices = find_nearest_indices(remaining_indices, centroids, pca, latents_matrix)

chosen_sample = None

for iteration in range(10):
    print(f"\nIteracja {iteration + 1}")
    current_points = [remaining_indices[idx] for idx in selected_indices]
    for i in range(len(current_points)):
        embedding = embeddings.iloc[current_points[i]]
        latent = latents.iloc[current_points[i]]
        folder_path = Path(f"/app/shared/krok{iteration}")
        folder_path.mkdir(parents=True, exist_ok=True)
        xtts_handler.inference(embedding, latent, f"/app/shared/krok{iteration}/test{i}.wav", " To jest napad! ")
    print("Wybrane punkty:", current_points)

# # WIZUALIZACJA
#     if len(remaining_indices) > 0:
#         group_latents_3d = pca.transform(latents_matrix[remaining_indices])
#
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#
#         ax.scatter(group_latents_3d[:, 0], group_latents_3d[:, 1], group_latents_3d[:, 2],
#                    alpha=0.4, label="Pozostałe punkty", color='blue')
#
#         selected_latents_3d = pca.transform(latents_matrix[current_points])
#         ax.scatter(selected_latents_3d[:, 0], selected_latents_3d[:, 1], selected_latents_3d[:, 2],
#                    color='red', s=300, label="Wybrane punkty", edgecolors='black', linewidths=2)
#
#         for i, idx in enumerate(current_points):
#             ax.text(selected_latents_3d[i, 0], selected_latents_3d[i, 1], selected_latents_3d[i, 2],
#                     f"{idx}", fontsize=10, color='black')
#
#         ax.set_title(f"Iteracja {iteration + 1} - Wybrane punkty na tle pozostałych")
#         ax.set_xlabel("Pierwsza składowa PCA")
#         ax.set_ylabel("Druga składowa PCA")
#         ax.set_zlabel("Trzecia składowa PCA")
#         ax.legend()
#         plt.show()

    user_input = int(input(f"Podaj indeks jednego z wybranych punktów {current_points}: "))
    if user_input not in current_points:
        print("Niepoprawny wybór. Spróbuj ponownie.")
        continue

    if iteration == 9:
        # Iteracja 10: wybór końcowy
        chosen_sample = user_input
        break

    # Odrzucamy 50% najdalszych punktów
    distances = cdist([latents_matrix[user_input]], latents_matrix[remaining_indices]).flatten()
    sorted_indices = np.argsort(distances)
    count = int(len(sorted_indices) * ratio)
    closest_indices = sorted_indices[:count]
    remaining_indices = [remaining_indices[i] for i in closest_indices]

    # Aktualizujemy PCA i centroidy
    pca = PCA(n_components=3)
    pca.fit(latents_matrix[remaining_indices])
    group_latents_3d = pca.transform(latents_matrix[remaining_indices])
    min_bounds = np.min(group_latents_3d, axis=0)
    max_bounds = np.max(group_latents_3d, axis=0)
    centroids = get_centroids(iteration+1, min_bounds, max_bounds)
    selected_indices = find_nearest_indices(remaining_indices, centroids, pca, latents_matrix)

if chosen_sample is not None:
    final_vector = latents.iloc[chosen_sample]
    final_path = paths.iloc[chosen_sample]
    print("\nWybrana próbka w 10 kroku:")
    print("Indeks próbki:", chosen_sample)
    print("Ścieżka pliku:", final_path)
else:
    print("Brak wybranej próbki w 10 kroku.")


# latent_matrix = np.array(latents.tolist())
# path_matrix = paths.tolist()

# def select_farthest_points_by_sum(distance_matrix, k=4):
#     """
#     Wybiera k najbardziej odległych punktów, maksymalizując sumę odległości.
#     """
#
#     # Zacznij od punktu najbardziej oddalonego od środka
#     initial_point = np.argmax(np.sum(distance_matrix, axis=0))
#     selected = [initial_point]
#
#     for _ in range(k - 1):
#         # Oblicz sumę odległości do już wybranych punktów
#         sum_distances = np.sum(distance_matrix[selected], axis=0)
#         next_point = np.argmax(sum_distances)  # Punkt o największej sumie odległości
#         selected.append(next_point)
#
#     return selected
#
# def iterative_search(embedding_matrix, k=4, steps=10):
#     """
#     Interaktywny algorytm zawężania próbek na podstawie wyborów użytkownika.
#
#     :param embedding_matrix: Macierz embeddingów (n próbek x wymiar embeddingu)
#     :param k: Liczba próbek prezentowanych w każdym kroku
#     :param steps: Maksymalna liczba kroków
#     :return: Indeks najbardziej dopasowanej próbki
#     """
#
#     n_samples = embedding_matrix.shape[0]
#     remaining_indices = list(range(n_samples))  # Zbiór wszystkich próbek
#
#     for step in range(steps):
#         # Wybierz k najbardziej rozproszonych próbek z pozostałych
#         if len(remaining_indices) <= k:
#             # Jeśli zostało mniej niż k próbek, zakończ iterację
#             print(f"Krok {step + 1}: Pozostałe próbki {remaining_indices}")
#             break
#
#         # Oblicz dystanse tylko między pozostałymi próbkami
#         remaining_embeddings = embedding_matrix[remaining_indices]
#         distances = cdist(remaining_embeddings, remaining_embeddings, metric="euclidean")
#
#         # Wybierz k najbardziej odległych próbek
#         selected_indices = select_farthest_points_by_sum(distances, k=k)
#         selected_samples = [remaining_indices[i] for i in selected_indices]
#
#         # Wyświetl próbki użytkownikowi i poproś o wybór
#         print(f"Krok {step + 1}: Wybrane próbki {selected_samples}")
#         chosen_sample = int(input(f"Podaj indeks wybranej próbki z {selected_samples}: "))
#
#         # Zawęź przestrzeń do najbliższych próbek w embeddingach
#         chosen_embedding = embedding_matrix[chosen_sample]
#         distances_to_chosen = cdist([chosen_embedding], embedding_matrix[remaining_indices], metric="euclidean")[0]
#         remaining_indices = [remaining_indices[i] for i in np.argsort(distances_to_chosen)[:len(remaining_indices) // 2]]
#
#     # Zwróć ostateczny wybór użytkownika
#     return chosen_sample
#
# iterative_search(latent_matrix, k=4, steps=5)

# latent_matrix = np.array(latents.tolist())
# path_matrix = paths.tolist()

# def select_farthest_points_by_sum(distance_matrix, k=4):
#     """
#     Wybiera k najbardziej odległych punktów, maksymalizując sumę odległości.
#     """
#
#     # Zacznij od punktu najbardziej oddalonego od środka
#     initial_point = np.argmax(np.sum(distance_matrix, axis=0))
#     selected = [initial_point]
#
#     for _ in range(k - 1):
#         # Oblicz sumę odległości do już wybranych punktów
#         sum_distances = np.sum(distance_matrix[selected], axis=0)
#         next_point = np.argmax(sum_distances)  # Punkt o największej sumie odległości
#         selected.append(next_point)
#
#     return selected
#
# def iterative_search(embedding_matrix, k=4, steps=10):
#     """
#     Interaktywny algorytm zawężania próbek na podstawie wyborów użytkownika.
#
#     :param embedding_matrix: Macierz embeddingów (n próbek x wymiar embeddingu)
#     :param k: Liczba próbek prezentowanych w każdym kroku
#     :param steps: Maksymalna liczba kroków
#     :return: Indeks najbardziej dopasowanej próbki
#     """
#
#     n_samples = embedding_matrix.shape[0]
#     remaining_indices = list(range(n_samples))  # Zbiór wszystkich próbek
#
#     for step in range(steps):
#         # Wybierz k najbardziej rozproszonych próbek z pozostałych
#         if len(remaining_indices) <= k:
#             # Jeśli zostało mniej niż k próbek, zakończ iterację
#             print(f"Krok {step + 1}: Pozostałe próbki {remaining_indices}")
#             break
#
#         # Oblicz dystanse tylko między pozostałymi próbkami
#         remaining_embeddings = embedding_matrix[remaining_indices]
#         distances = cdist(remaining_embeddings, remaining_embeddings, metric="euclidean")
#
#         # Wybierz k najbardziej odległych próbek
#         selected_indices = select_farthest_points_by_sum(distances, k=k)
#         selected_samples = [remaining_indices[i] for i in selected_indices]
#
#         # Wyświetl próbki użytkownikowi i poproś o wybór
#         print(f"Krok {step + 1}: Wybrane próbki {selected_samples}")
#         chosen_sample = int(input(f"Podaj indeks wybranej próbki z {selected_samples}: "))
#
#         # Zawęź przestrzeń do najbliższych próbek w embeddingach
#         chosen_embedding = embedding_matrix[chosen_sample]
#         distances_to_chosen = cdist([chosen_embedding], embedding_matrix[remaining_indices], metric="euclidean")[0]
#         remaining_indices = [remaining_indices[i] for i in np.argsort(distances_to_chosen)[:len(remaining_indices) // 2]]
#
#     # Zwróć ostateczny wybór użytkownika
#     return chosen_sample
#
# iterative_search(latent_matrix, k=4, steps=5)
