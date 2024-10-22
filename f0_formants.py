import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# Funkcja do obliczenia f0
def calculate_pitch(sound):
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]  # Pominięcie wartości zerowych
    if len(f0_values) > 0:
        return np.mean(f0_values)  # Zwraca średnią wartość f0
    else:
        return None

# Funkcja do obliczenia formantów
def calculate_formants(sound):
    formant = sound.to_formant_burg()  # Wykorzystuje metodę Burg do analizy formantów
    formants = {}
    for t in np.linspace(0, sound.get_total_duration(), num=100):  # Próbkowanie 100 punktów w czasie
        formants_at_time = [formant.get_value_at_time(i, t) for i in range(1, 5)]  # Formanty F1-F4
        formants[t] = formants_at_time
    return formants

# Funkcja do rysowania formantów
def plot_formants(formants):
    times = list(formants.keys())
    formant_values = np.array(list(formants.values()))

    for i in range(formant_values.shape[1]):
        plt.plot(times, formant_values[:, i], label=f'F{i+1}')

    plt.xlabel("Czas (s)")
    plt.ylabel("Częstotliwość (Hz)")
    plt.legend()
    plt.title("Formanty F1-F4")
    plt.show()

# Wczytanie próbki głosu
def analyze_voice(file_path):
    sound = parselmouth.Sound(file_path)

    # Obliczanie f0
    f0 = calculate_pitch(sound)
    if f0:
        print(f"Średnia f0: {f0:.2f} Hz")
    else:
        print("Nie udało się obliczyć f0")

    # Obliczanie formantów
    formants = calculate_formants(sound)
    plot_formants(formants)

# Przykład użycia
file_path1 = "Fusion power plant plant reaches major milestone.wav" #Średnia f0: 112.89 Hz
file_path2 = "10 films that could win Oscars in 2025.wav" #Średnia f0: 121.85 Hz
file_path3 = "Amazon cuts hundreds of jobs in cloud business.wav" #Średnia f0: 111.25 Hz
analyze_voice(file_path3)
