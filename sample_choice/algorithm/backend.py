import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from pathlib import Path
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, \
    LATENT_KEY
from settings import SAMPLE_PATH_DB_KEY
from utils.xtts_handler import XTTSHandler
from sample_choice.algorithm.gender_list import man_indexes, women_indexes

class BackendEngine:
    def __init__(self):
        self.edb = EmbeddingDatabaseManager()

        self.embeddings = self.edb.get_all_values_from_column(EMBEDDING_KEY)
        self.latents = self.edb.get_all_values_from_column(LATENT_KEY)
        self.paths = self.edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)

        # Konwersja do macierzy NumPy
        self.latents_matrix = np.vstack(self.latents.values)

        # Indeksy mężczyzn i kobiet (podane przez użytkownika)
        self.man_indexes = man_indexes
        self.women_indexes = women_indexes

    def initialize_for_group(self, group_choice):
        """
        Inicjalizuje PCA i dane wejściowe dla wybranej grupy (m/k).
        Zwraca ratio oraz remaining_indices.
        """
        if group_choice == 'm':
            remaining_indices = self.man_indexes.copy()
            ratio = 0.5
        elif group_choice == 'k':
            remaining_indices = self.women_indexes.copy()
            ratio = 0.65
        else:
            raise ValueError("Niepoprawny wybór. Wybierz 'm' lub 'k'.")

        # Dopasowanie PCA tylko do wybranej grupy
        self.pca = PCA(n_components=3)
        self.pca.fit(self.latents_matrix[remaining_indices])
        self.group_latents_3d = self.pca.transform(self.latents_matrix[remaining_indices])
        self.remaining_indices = remaining_indices
        self.min_bounds = np.min(self.group_latents_3d, axis=0)
        self.max_bounds = np.max(self.group_latents_3d, axis=0)
        self.ratio = ratio
        return remaining_indices, ratio

    def get_centroids(self, iteration, min_b, max_b):
        mid_x = (min_b[0] + max_b[0]) / 2
        mid_y = (min_b[1] + max_b[1]) / 2
        mid_z = (min_b[2] + max_b[2]) / 2

        if iteration == 0:
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
            centroids = [
                [min_b[0], min_b[1], mid_z],
                [min_b[0], max_b[1], mid_z],
                [mid_x, min_b[1], mid_z],
                [mid_x, max_b[1], mid_z],
                [max_b[0], min_b[1], mid_z],
                [max_b[0], max_b[1], mid_z],
            ]
        elif 2 <= iteration <= 6:
            centroids = [
                [mid_x, min_b[1], min_b[2]],
                [mid_x, max_b[1], min_b[2]],
                [mid_x, min_b[1], max_b[2]],
                [mid_x, max_b[1], max_b[2]],
            ]
        elif 7 <= iteration <= 8:
            centroids = [
                [min_b[0], mid_y, mid_z],
                [mid_x, max_b[1], mid_z],
                [max_b[0], mid_y, mid_z],
            ]
        else:
            centroids = [
                [mid_x, min_b[1], mid_z],
                [mid_x, max_b[1], mid_z],
            ]
        return centroids

    def find_nearest_indices(self, remaining_indices, centroids):
        remaining_latents_3d = self.pca.transform(self.latents_matrix[remaining_indices])
        selected_indices = []
        for centroid in centroids:
            distances = cdist(remaining_latents_3d, [centroid])
            nearest_index = np.argmin(distances)
            selected_indices.append(nearest_index)
        return selected_indices

    def run_inference_for_step(self, iteration, user_text):
        """
        Uruchamia wybór próbek dla danego kroku (iteration),
        generuje pliki wav i zwraca listę wybranych indeksów.
        """
        centroids = self.get_centroids(iteration, self.min_bounds, self.max_bounds)
        selected_indices_local = self.find_nearest_indices(self.remaining_indices, centroids)
        current_points = [self.remaining_indices[idx] for idx in selected_indices_local]

        folder_path = Path(f"/app/shared/krok{iteration + 1}")
        folder_path.mkdir(parents=True, exist_ok=True)

        for i, cp in enumerate(current_points):
            embedding = self.embeddings.iloc[cp]
            latent = self.latents.iloc[cp]
            # Inference z użyciem podanego tekstu
            self.xtts_handler = XTTSHandler()
            self.xtts_handler.inference(embedding, latent, f"/app/shared/krok{iteration + 1}/test{i}.wav", user_text)

        return current_points
