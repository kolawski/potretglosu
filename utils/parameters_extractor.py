import crepe
import librosa
import numpy as np
import parselmouth
import scipy.stats
from scipy.io import wavfile
from transformers import pipeline

from utils.audio_sample_converter import audio_to_vec_librosa, normalize_audio
from utils.exceptions import VoiceModifierError

CONFIDENCE_THRESHOLD = 0.5
MODEL_CAPACITY = 'small'
STEP_SIZE = 300

ROLL_PERCENT=0.85
PROCESS_POINT_MINIMUM_PITCH = 75
PROCESS_POINT_MAXIMUM_PITCH = 500

F0_KEY = 'f0'
GENDER_KEY = 'male'
VARIANCE_KEY = 'variance'
SKEWNESS_KEY = 'skewness'
KURTOSIS_KEY = 'kurtosis'
INTENSITY_KEY = 'intensity'
JITTER_KEY = 'jitter'
SHIMMER_KEY = 'shimmer'
HNR_KEY = 'hnr'
ZERO_CROSSING_RATE_KEY = 'zero_crossing_rate'
SPECTRAL_CENTROID_KEY = 'spectral_centroid'
SPECTRAL_BANDWIDTH_KEY = 'spectral_bandwidth'
SPECTRAL_FLATNESS_KEY = 'spectral_flatness'
SPECTRAL_ROLL_OFF_KEY = 'spectral_roll_off'
TONNETZ_KEY = 'tonnetz'
TEMPO_KEY = 'tempo'


class ParametersExtractor:
    def __init__(self):
        #TODO docstringi
        crepe.core.build_and_load_model(MODEL_CAPACITY)
        self._file_path = None
        self._sr, self._wavfile_audio = None, None
        self._librosa_audio = None
        self._parselmouth_audio = None
        self._point_process = None
        self._gender_pipe = pipeline("audio-classification", model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")

    def _initialize_audio_sample(self, file_path):
        self._file_path = file_path
        self._sr, self._wavfile_audio = wavfile.read(file_path)
        librosa_audio, _ = audio_to_vec_librosa(file_path)
        self._librosa_audio = normalize_audio(librosa_audio)
        self._parselmouth_audio = parselmouth.Sound(file_path)

    def _deinitialize_audio_sample(self):
        self._sr, self._wavfile_audio, self._file_path = None, None, None
        self._librosa_audio = None
        self._parselmouth_audio = None
        self._point_process = None

    def ensure_audio_sample_initialized(func):
        def wrapper(self, *args, **kwargs):
            if self._sr is None or self._wavfile_audio is None:
                raise VoiceModifierError("Audio sample is not initialized.")
            return func(self, *args, **kwargs)
        return wrapper
    
    def initialize_and_deinitialize_audio_sample(func):
        def wrapper(self, file_path, *args, **kwargs):
            self._initialize_audio_sample(file_path)
            try:
                result = func(self, file_path, *args, **kwargs)
            finally:
                self._deinitialize_audio_sample()
            return result
        return wrapper

    @ensure_audio_sample_initialized
    def predict_f0(self):
        time, frequency, confidence, activation = \
            crepe.predict(self._wavfile_audio, self._sr, viterbi=True, step_size=STEP_SIZE, model_capacity=MODEL_CAPACITY)
        # crepe.process_file(file_path, output='/app/results', viterbi=True, step_size=STEP_SIZE, model_capacity=MODEL_CAPACITY, save_plot=True, plot_voicing=True) 

        confident_frequencies = [x for x, y in zip(frequency, confidence) if y >= CONFIDENCE_THRESHOLD]

        if confident_frequencies:
            average_frequency = round(sum(confident_frequencies) / len(confident_frequencies), 2)
            return average_frequency
        else:
            return None

    # needs ffmpeg
    @ensure_audio_sample_initialized
    def get_gender(self):
        result = self._gender_pipe(self._file_path)
        return round(100 * next(item['score'] for item in result if item['label'] == 'male'), 2)

    @ensure_audio_sample_initialized
    def get_variance(self):
        return round(np.var(self._librosa_audio), 4)
    
    @ensure_audio_sample_initialized
    def get_skewness(self):
        return round(scipy.stats.skew(self._librosa_audio), 4)
    
    @ensure_audio_sample_initialized
    def get_kurtosis(self):
        return round(scipy.stats.kurtosis(self._librosa_audio), 4)
    
    @ensure_audio_sample_initialized
    def get_intensity(self):
        return round(self._parselmouth_audio.to_intensity().get_average(), 4)
    
    @ensure_audio_sample_initialized
    def get_jitter(self):
        if self._point_process is None:
            self._point_process = parselmouth.praat.call(self._parselmouth_audio, "To PointProcess (periodic, cc)...", PROCESS_POINT_MINIMUM_PITCH, PROCESS_POINT_MAXIMUM_PITCH)
        return round(parselmouth.praat.call(self._point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 4)
    
    @ensure_audio_sample_initialized
    def get_shimmer(self):
        if self._point_process is None:
            self._point_process = parselmouth.praat.call(self._parselmouth_audio, "To PointProcess (periodic, cc)...", PROCESS_POINT_MINIMUM_PITCH, PROCESS_POINT_MAXIMUM_PITCH)
        return round(parselmouth.praat.call([self._parselmouth_audio, self._point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 4)
    
    @ensure_audio_sample_initialized
    def get_hnr(self):
        harmonicity = parselmouth.praat.call(self._parselmouth_audio, "To Harmonicity (cc)", 0.01, PROCESS_POINT_MINIMUM_PITCH, 0.1, 1.0)
        return round(parselmouth.praat.call(harmonicity, "Get mean", 0, 0), 4)
    
    @ensure_audio_sample_initialized
    def get_zero_crossing_rate(self):
        return round(np.sum(np.abs(np.diff(np.sign(self._librosa_audio)))) / len(self._librosa_audio), 4)
    
    @ensure_audio_sample_initialized
    def get_spectral_centroid(self):
        return round(np.mean(librosa.feature.spectral_centroid(y=self._librosa_audio, sr=self._sr), axis=1)[0], 4)

    @ensure_audio_sample_initialized
    def get_spectral_bandwidth(self):
        return round(np.mean(librosa.feature.spectral_bandwidth(y=self._librosa_audio, sr=self._sr), axis=1)[0], 4)
    
    @ensure_audio_sample_initialized
    def get_spectral_flatness(self):
        return round(np.mean(librosa.feature.spectral_flatness(y=self._librosa_audio), axis=1)[0], 4)

    @ensure_audio_sample_initialized
    def get_spectral_roll_off(self):
        return round(np.mean(librosa.feature.spectral_rolloff(y=self._librosa_audio, sr=self._sr, roll_percent=ROLL_PERCENT), axis=1)[0], 4)

    # needs numpy 1.23.0
    @ensure_audio_sample_initialized
    def get_tonnetz(self):
        harmonic = librosa.effects.harmonic(self._librosa_audio)
        return round(np.mean(librosa.feature.tonnetz(y=harmonic, sr=self._sr), axis=1)[0], 4)

    # needs numpy 1.23.0
    @ensure_audio_sample_initialized
    def get_tempo(self):
        onset_env = librosa.onset.onset_strength(y=self._librosa_audio, sr=self._sr)
        return librosa.feature.tempo(onset_envelope=onset_env, sr=self._sr)[0]

    @initialize_and_deinitialize_audio_sample
    def extract_parameters(self, file_path):
        results = {}
        results[F0_KEY] = self.predict_f0()
        results[GENDER_KEY] = self.get_gender()
        results[VARIANCE_KEY] = self.get_variance()
        results[SKEWNESS_KEY] = self.get_skewness()
        results[KURTOSIS_KEY] = self.get_kurtosis()
        results[INTENSITY_KEY] = self.get_intensity()
        results[JITTER_KEY] = self.get_jitter()
        results[SHIMMER_KEY] = self.get_shimmer()
        results[HNR_KEY] = self.get_hnr()
        results[ZERO_CROSSING_RATE_KEY] = self.get_zero_crossing_rate()
        results[SPECTRAL_CENTROID_KEY] = self.get_spectral_centroid()
        results[SPECTRAL_BANDWIDTH_KEY] = self.get_spectral_bandwidth()
        results[SPECTRAL_FLATNESS_KEY] = self.get_spectral_flatness()
        results[SPECTRAL_ROLL_OFF_KEY] = self.get_spectral_roll_off()
        results[TONNETZ_KEY] = self.get_tonnetz()
        results[TEMPO_KEY] = self.get_tempo()
        return results
