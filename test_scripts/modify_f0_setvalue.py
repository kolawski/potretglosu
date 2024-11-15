import librosa
import soundfile as sf
import numpy as np
# TRZEBA WYSKALOWAĆ
# Wczytaj plik audio
filename = '/app/Resources/different_f0_test/original_20606171.wav'
y, sr = librosa.load(filename)

# Ustal docelową wartość f0 w Hz (np. 300 Hz)
target_f0 = 600
# Wyznacz bieżące f0 próbki
f0_estimation = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_mean = np.mean(f0_estimation[f0_estimation > 0])  # Uwzględnia tylko dodatnie wartości
print(f"Średnia f0: {f0_mean}")

# Oblicz liczbę półtonów na podstawie stosunku target_f0 do f0_mean
if f0_mean > 0:
    n_steps = 12 * np.log2(target_f0 / f0_mean)
else:
    raise ValueError("Nie udało się poprawnie określić f0 oryginału")

# Zmień wysokość dźwięku bez zmiany długości trwania próbki
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# Zapisz wynik
output_filename = '/app/Resources/different_f0_test/changed.wav'
sf.write(output_filename, y_shifted, sr)

print(f"Nowy plik zapisano jako {output_filename}, z docelowym f0 bliskim {target_f0} Hz")
