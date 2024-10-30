import librosa
import soundfile as sf

file = 'Resources/audio_samples/10 films that could win Oscars in 2025.wav'  # Zamień na ścieżkę do swojego pliku audio
y, sr = librosa.load(file)

y_shifted = librosa.effects.pitch_shift(y, n_steps=4, sr=sr) # greater steps, higher f0
y_streched = librosa.effects.time_stretch(y_shifted, rate=2.0) # greater rate, faster audio

output_file = 'Resources/audio_samples/Modified 10 films that could win Oscars in 2025.wav'
sf.write(output_file, y_shifted, sr)