import librosa
import numpy as np
import soundfile as sf

file = '/app/Resources/different_f0_test/original_20606171.wav'  # Zamień na ścieżkę do swojego pliku audio
y, sr = librosa.load(file, sr=48000)
print(y.shape)
# y = y.astype(np.float32)

y_shifted = librosa.effects.pitch_shift(y, n_steps=4, sr=48000) # greater steps, higher f0
# y_streched = librosa.effects.time_stretch(y_shifted, rate=2.0) # greater rate, faster audio

output_file = '/app/Resources/different_f0_test/changed.wav'
sf.write(output_file, y_shifted, sr)