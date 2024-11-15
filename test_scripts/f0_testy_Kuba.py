import librosa
import librosa.display
import numpy as np
import soundfile as sf

# Wczytaj plik audio
dir = "/app/Resources/different_f0_test/"
audio_path = dir +'original_20606171.wav'
y, sr = librosa.load(audio_path)
y = y.astype(np.float32)
# Zdefiniuj zakres f0 i deltę
f0_min = 100  # minimalna częstotliwość f0
f0_max = 400  # maksymalna częstotliwość f0
delta_f0 = 40  # zmiana f0

def change_f0(y, sr, target_f0):
    # Analiza f0
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    # Wybieramy częstotliwości f0 - dla uproszczenia użyjemy średniej
    f0_mean = np.mean(pitches[pitches > 0])
    print(f0_mean)
    
    # Obliczamy, o ile semitonów trzeba podnieść częstotliwość
    semitones = 12 * np.log2(target_f0 / f0_mean)
    
    # Zmiana f0 (pitch shift)
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)
    
    return y_shifted

# Generowanie próbek w zadanym zakresie
for f0_target in range(f0_min, f0_max + 1, delta_f0):
    # Zmieniamy f0 w próbce
    y_modified = change_f0(y, sr, f0_target)
    
    # Zapisz próbkę do pliku
    output_filename = f'{dir}output_f0_{f0_target}Hz.wav'
    sf.write(output_filename, y_modified, sr)
    print(f"Zapisano plik: {output_filename}")

import librosa
import numpy as np
import scipy.stats as stats
import scipy.signal

class AudioAnalyzer:
    def __init__(self, audio_path, sr=22050):
        self.audio, self.sr = librosa.load(audio_path, sr=sr)
        self.f0 = librosa.pyin(self.audio, fmin=50, fmax=500, sr=self.sr)[0]
        self.formants = None  # Ustalimy później podczas analizy formantów

    def max_f0(self):
        return np.nanmax(self.f0)

    def min_f0(self):
        return np.nanmin(self.f0)

    def mean_f0(self):
        return np.nanmean(self.f0)

    def calculate_formants(self):
        """Analiza formantów i antyformantów z wykorzystaniem LPC."""
        formants, _ = librosa.lpc(self.audio, order=8)  # Przykładowy rząd LPC, można zmieniać
        self.formants = formants
        return formants

    def f1(self):
        if self.formants is None:
            self.calculate_formants()
        return self.formants[0]

    def f2(self):
        if self.formants is None:
            self.calculate_formants()
        return self.formants[1]

    def f3(self):
        if self.formants is None:
            self.calculate_formants()
        return self.formants[2]

    def antiformants(self):
        """Znajduje antyformanty (przypuszczenie: minima w widmie)"""
        minima = scipy.signal.argrelextrema(self.audio, np.less)
        return minima

    def energy(self):
        return np.sum(self.audio ** 2) / len(self.audio)

    def mean_power(self):
        return np.mean(self.audio ** 2)

    def central_moment(self, order):
        """Moment centralny z normalizacją."""
        return stats.moment(self.audio, moment=order)

    def skewness(self):
        return stats.skew(self.audio)

    def kurtosis(self):
        return stats.kurtosis(self.audio)

    def jitter(self):
        """Jitter obliczony jako różnica między częstotliwościami f0."""
        diffs = np.diff(self.f0[~np.isnan(self.f0)])
        return np.mean(np.abs(diffs))

    def shimmer(self):
        """Shimmer obliczony jako różnica między amplitudami."""
        amplitude_diffs = np.diff(np.abs(self.audio))
        return np.mean(np.abs(amplitude_diffs))

# Przykład użycia:
dir_path = "/app/Resources/ready_audio_samples/"
file_path1 = f"{dir_path}a_common_voice_pl_21643510.wav" #Średnia f0: 112.89 Hz
file_path2 = f"{dir_path}b_common_voice_pl_20606171.wav" #Średnia f0: 121.85 Hz
file_path3 = f"{dir_path}c_common_voice_pl_20613853.wav"
analyzer = AudioAnalyzer(file_path1)
print("Max f0:", analyzer.max_f0())
print("Min f0:", analyzer.min_f0())
print("Średnie f0:", analyzer.mean_f0())
print("F1:", analyzer.f1())
print("F2:", analyzer.f2())
print("F3:", analyzer.f3())
print("Antyformanty:", analyzer.antiformants())
print("Energia:", analyzer.energy())
print("Moc średnia:", analyzer.mean_power())
print("Moment zerowy rzędu 1:", analyzer.central_moment(1))
print("Moment zerowy rzędu 2:", analyzer.central_moment(2))
print("Moment zerowy rzędu 3:", analyzer.central_moment(3))
print("Moment zerowy rzędu 4:", analyzer.central_moment(4))
print("Skośność:", analyzer.skewness())
print("Kurtoza:", analyzer.kurtosis())
print("Jitter:", analyzer.jitter())
print("Shimmer:", analyzer.shimmer())



# def extract_parameters(file_path):
#     """
#     Extracts parameters from a file path.

#     :param file_path: path to a file
#     :type file_path: str
#     :return: parameters extracted from the file path
#     :rtype: dict
#     """
#     return {
#         "file_name": Path(file_path).name,
#         "file_stem": Path(file_path).stem,
#         "file_suffix": Path(file_path).suffix,
#         "file_parent": Path(file_path).parent,
#         "file_parts": Path(file_path).parts,
#         "file_drive": Path(file_path).drive,
#         "file_anchor": Path(file_path).anchor,
#         "file_as_posix": Path(file_path).as_posix(),
#     }


