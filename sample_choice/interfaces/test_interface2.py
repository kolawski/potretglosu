import math
from database_management.database_managers.embedding_database_manager import (
    EmbeddingDatabaseManager,
    EMBEDDING_KEY,
    LATENT_KEY,
    SAMPLE_PATH_DB_KEY,
)
from sample_choice.algorithm.gender_list import man_indexes, women_indexes
from utils.xtts_handler import XTTSHandler
from pathlib import Path
from utils.metrics_retriever import retrieve_metrics
from settings import LATENT_SHAPE, EMBEDDING_SHAPE
from utils.embedding_converter import flat_to_torch

xtts_handler = XTTSHandler()
edb = EmbeddingDatabaseManager()
embeddings = edb.get_all_values_from_column(EMBEDDING_KEY)
latents = edb.get_all_values_from_column(LATENT_KEY)
paths = edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)

latent1, embedding1 = xtts_handler.compute_latents("/app/shared/speakers/speaker_1_30s.wav")
"""
latent2, embedding2 = xtts_handler.compute_latents("/app/shared/speakers/speaker_2_30s_women.wav")
latent3, embedding3 = xtts_handler.compute_latents("/app/shared/speakers/speaker_3_30s.wav")
latent4, embedding4 = xtts_handler.compute_latents("/app/shared/speakers/speaker_4_40s.wav")
latent5, embedding5 = xtts_handler.compute_latents("/app/shared/speakers/speaker_5_20s.wav")
"""

def choose_n_points_farthest(points, subset_indexes, n):
    """
    Wybiera n punktów daleko od siebie (farthest point sampling),
    ale tylko w obrębie podzbioru indeksów subset_indexes.

    Parametry:
    -----------
    points : list
        Lista "globalnych" punktów (np. tuple lub np.array),
        gdzie points[i] to wektor (embedding).
    subset_indexes : list[int]
        Lista indeksów globalnych, w których chcemy szukać najdalszych punktów.
    n : int
        Liczba punktów do wybrania.

    Zwraca:
    -----------
    chosen_global_idx : list[int]
        Lista n indeksów globalnych (spośród subset_indexes),
        które są od siebie najdalsze.
    """
    # 1. Tworzymy "lokalny" zbiór punktów - tylko z subset_indexes
    points_subset = [points[i] for i in subset_indexes]
    N = len(points_subset)

    if n < 2:
        raise ValueError("n musi być co najmniej 2.")
    if n > N:
        raise ValueError("n nie może być większe niż liczba dostępnych punktów w podzbiorze.")

    # a) Znajdź najbardziej oddaloną parę w 'points_subset'
    max_dist = -1
    p1_idx, p2_idx = None, None
    for i in range(N):
        for j in range(i + 1, N):
            dist = retrieve_metrics(points_subset[i], points_subset[j])
            if dist > max_dist:
                max_dist = dist
                p1_idx, p2_idx = i, j

    chosen_local_idx = [p1_idx, p2_idx]

    # b) Dodawaj kolejne punkty (max-min distance)
    while len(chosen_local_idx) < n:
        best_candidate_idx = None
        best_min_dist = -1

        for k in range(N):
            if k in chosen_local_idx:
                continue
            d_min = min(retrieve_metricspoints_subset[k], points_subset[idx])
                        for idx in chosen_local_idx)
            if d_min > best_min_dist:
                best_min_dist = d_min
                best_candidate_idx = k

        chosen_local_idx.append(best_candidate_idx)

    # 3. Mapujemy indeksy lokalne (w podzbiorze) na indeksy globalne
    chosen_global_idx = [subset_indexes[i] for i in chosen_local_idx]
    return chosen_global_idx


if __name__ == "__main__":
    # Konwersja wektorów do tuple (by uniknąć np. problemów z `pt in chosen` dla np.array)

    points = [row for row in latents]

    indexes = []
    ratio = 0
    gender = input("Podaj płeć (m - mężczyzna, k - kobieta): ")

    if gender == 'm':
        indexes = man_indexes
        ratio = 0.5
    elif gender == 'k':
        indexes = women_indexes
        ratio = 0.65

    last_choice = 0  # Tutaj będziemy zapamiętywać ostatnio wybrany indeks
    euclidean_distance_embedding = []
    euclidean_distance_latent = []
    selected_distance_embedding = []
    selected_distance_latent = []

    input_text = input("Wprowadź zapamiętaną wypowiedź: ")

    # Powtarzamy 10 iteracji (lub do wyczerpania / przerwania)
    for step in range(10):
        if step == 0:
            n = 8
        elif step == 1 or step == 2:
            n = 6
        elif step >= 3 and step <= 6:
            n = 4
        elif step == 7 and step == 8:
            n = 3
        else:
            n = 2
        # 2. Wybierz n punktów daleko od siebie
        selected_idx = choose_n_points_farthest(points, indexes, n)
        for i in range(len(selected_idx)):
            embedding = embeddings.iloc[selected_idx[i]]
            latent = latents.iloc[selected_idx[i]]

            print(f"embedding size: {embedding.size}")
            print(f"embedding1 size: {embedding1.size}")

            print(f"embedding shape: {embedding.shape}")
            print(f"embedding1 shape: {embedding1.shape}")

            print(f"embedding type: {type(embedding)}")
            print(f"embedding1 type: {type(embedding1)}")


            folder_path = Path(f"/app/shared/krok{step}")
            folder_path.mkdir(parents=True, exist_ok=True)
            xtts_handler.inference(embedding, latent, f"/app/shared/krok{step+1}/Nagranie{i+1} - indeks {selected_idx[i]}.wav", input_text)
            print(embedding.size)
            print(embedding1.size)
            euclidean_distance_embedding.append(retrieve_metrics(embedding, embedding1))
            euclidean_distance_latent.append(retrieve_metrics(latent, latent1))
        print(f"\nKrok {step + 1}. Wybrano punkty o indeksach: {selected_idx}")

        choice = int(input("Podaj WYBRANY INDEKS (z powyższej listy): "))

        last_choice = choice  # Zapamiętujemy go
        selected_distance_embedding.append(retrieve_metrics(embeddings.iloc[choice], embedding1))
        selected_distance_latent.append(retrieve_metrics(latents.iloc[choice], embedding1))

        # 4. Liczymy odległość od choice do wszystkich próbek w 'indexes'
        dist_list = []
        for idx in indexes:
            d = math.dist(points[choice], points[idx])
            dist_list.append((idx, d))

        # 5. Sortujemy rosnąco po odległości (najbliższe na początku)
        dist_list.sort(key=lambda x: x[1])

        # 6. Zachowujemy ratio% najbliższych
        keep_count = int(len(indexes) * ratio)
        dist_list = dist_list[:keep_count]  # bierzemy tylko X% (np. 50%, 65%)

        # 7. Aktualizujemy 'indexes' — tylko te najbliższe
        indexes = [item[0] for item in dist_list]
        print(f"Zachowano {len(indexes)} indeksów (czyli ~{100 * ratio:.0f}%).\n")

    # Po zakończeniu pętli (10 iteracji lub przerwanie)
    print("Koniec pętli.")
    if last_choice is not None:
        print(f"Ostatnio wybrany indeks: {last_choice}")
        print(f"Odległości - wybrane embeddingi: {selected_distance_embedding}")
        print(f"Odległości - wybrane latenty: {selected_distance_latent}")
        print(f"Odległości - wszystkie embeddingi: {selected_distance_embedding}")
        print(f"Odległości - wszystkie latenty: {selected_distance_latent}")
    else:
        print("Nie wybrano żadnego indeksu w trakcie pętli.")
