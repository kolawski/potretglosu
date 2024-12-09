from data_loader import get_initial_samples
from scipy.spatial.distance import cdist
import numpy as np
from pydub import AudioSegment
import os

# Katalog współdzielony dla próbek audio
SHARED_DIR = "/app/shared"

def play_audio_on_host(file_path):
    """
    Wywołuje odtwarzanie pliku audio na hoście za pomocą domyślnego odtwarzacza.
    """
    print(f"Odtwarzanie pliku: {file_path}")
    os.system(f"start {file_path}")  # Dla Windows

def play_audio_in_container(file_path):
    """
    Odtwarza plik audio w kontenerze za pomocą ffplay.
    """
    try:
        print(f"Odtwarzanie pliku: {file_path}")
        # Dodanie opcji `-nodisp` i `-autoexit` dla prostszego odtwarzania
        os.system(f"ffplay -nodisp -autoexit {file_path}")
    except Exception as e:
        print(f"Błąd podczas odtwarzania pliku: {e}")


def interactive_playback(exported_files):
    """
    Interaktywne odtwarzanie plików przez użytkownika z możliwością wyboru najlepszej próbki.
    """
    chosen_sample = None

    while True:
        print("\nPliki dostępne do odtwarzania:")
        for idx, file_name in enumerate(exported_files):
            print(f"{idx + 1}. {file_name}")
        print("5. Przejdź do wyboru próbki i następnego kroku")

        try:
            choice = int(input("Wpisz numer pliku do odtworzenia (1-5): "))
            if choice == 5:
                if chosen_sample is not None:
                    return chosen_sample
                else:
                    print("Nie wybrano żadnej próbki. Najpierw odsłuchaj i wybierz numer.")
            elif 1 <= choice <= len(exported_files):
                play_audio_in_container(f"/app/shared/{exported_files[choice - 1]}")
                chosen_sample = choice - 1  # Zapisz ostatnio odtwarzany numer
            else:
                print("Nieprawidłowy wybór. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowy wybór. Wprowadź liczbę całkowitą.")




def export_audio_for_host(file_path, output_name):
    """
    Eksportuje plik audio do katalogu współdzielonego dla hosta.
    """
    try:
        output_path = os.path.join(SHARED_DIR, output_name)
        audio = AudioSegment.from_file(file_path)
        audio.export(output_path, format="wav")
        print(f"Plik zapisany w: {output_path}")
        return output_path
    except Exception as e:
        print(f"Błąd podczas eksportu pliku: {e}")
        return None


def play_audio_instructions():
    """
    Wyświetla instrukcje odsłuchu próbek na hoście.
    """
    print("\nInstrukcje:")
    print(f"1. Przejdź do katalogu 'shared' na swoim hoście.")
    print(f"2. Odtwórz pliki audio (np. 'sample_0.wav', 'sample_1.wav', ...).")
    print("3. Wybierz numer próbki najlepiej pasującej do Twojej pamięci dźwięku.\n")


def iterative_search(embedding_matrix, paths, k=4, max_steps=5):
    """
    Interaktywny algorytm wyboru najlepszego dźwięku na podstawie wyborów użytkownika.

    :param embedding_matrix: Macierz embeddingów (n próbek x wymiar embeddingu)
    :param paths: Ścieżki do plików audio
    :param k: Liczba próbek w każdym kroku
    :param max_steps: Maksymalna liczba kroków
    """
    remaining_indices = list(range(len(embedding_matrix)))  # Indeksy wszystkich dostępnych próbek

    for step in range(max_steps):
        print(f"\n=== KROK {step + 1} ===")

        # Jeśli mniej próbek niż k, zakończ proces
        if len(remaining_indices) <= k:
            print(f"Tylko {len(remaining_indices)} próbki pozostały, zakończenie procesu.")
            break

        # Wybierz k próbek do odsłuchania
        selected_indices = np.random.choice(remaining_indices, k, replace=False)
        selected_files = [paths[i] for i in selected_indices]

        # Eksport próbek do katalogu współdzielonego
        exported_files = []
        for idx, file_path in enumerate(selected_files):
            output_name = f"sample_{idx}.wav"
            export_audio_for_host(file_path, output_name)
            exported_files.append(output_name)

        # Wyświetl instrukcje dla użytkownika
        play_audio_instructions()

        # Interaktywne odtwarzanie i wybór próbki
        chosen_sample = interactive_playback(exported_files)
        chosen_index = selected_indices[chosen_sample]

        # Zaktualizuj listę pozostałych próbek
        chosen_embedding = embedding_matrix[chosen_index]
        remaining_embeddings = np.array([embedding_matrix[i] for i in remaining_indices])
        distances = cdist([chosen_embedding], remaining_embeddings, metric="euclidean")[0]

        # Zachowaj tylko najbliższe próbki
        half_size = len(remaining_indices) // 2
        nearest_indices = np.argsort(distances)[:half_size]
        remaining_indices = [remaining_indices[i] for i in nearest_indices]

        print(f"Wybrano próbkę: {paths[chosen_index]}")

    # Zwróć ostatecznie wybraną próbkę
    final_index = remaining_indices[0]
    print(f"\nNajlepsza dopasowana próbka: {paths[final_index]}")
    return final_index



if __name__ == "__main__":
    # Pobierz początkowe próbki
    paths, indices, embedding_matrix = get_initial_samples()

    # Wyświetl wybrane próbki
    print("Startowe próbki:")
    for idx in indices:
        print(f"Indeks: {idx}, Ścieżka: {paths[idx]}")

    # Upewnij się, że katalog współdzielony istnieje
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    # Przeprowadź iteracyjne wyszukiwanie
    iterative_search(embedding_matrix, paths)
