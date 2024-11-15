from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from utils.exceptions import VoiceModifierError

def audio_to_vec_librosa(sample_path):
    """Converts audio file to a vector using librosa library

    :param sample_path: path to an audio file
    :type sample_path: str
    :return: audio vector and sample rate
    :rtype: tuple of np.ndarray and int
    """
    if not Path(sample_path).is_file():
        raise VoiceModifierError(f"Audio file {sample_path} does not exist")
    return librosa.load(sample_path, sr=None)

def normalize_audio(audio):
    """Normalizes the audio signal to the range [-1, 1].

    :param audio: The input audio signal as a NumPy array.
    :type audio: numpy.ndarray
    :return: The normalized audio signal.
    :rtype: numpy.ndarray
    """
    return audio / np.max(np.abs(audio))
    
def vec_to_audio_librosa(audio_vector, sr, path_to_save):
    """Converts an audio vector to an audio file and saves it using the librosa library.

    :param audio_vector: The audio data to be saved.
    :type audio_vector: numpy.ndarray
    :param sr: The sample rate of the audio data.
    :type sr: int
    :param path_to_save: The file path where the audio file will be saved.
    :type path_to_save: str
    :returns: None
    """
    if Path(path_to_save).is_file():
        print(f"Audio file {path_to_save} exists, overwriting")
    sf.write(path_to_save, audio_vector, sr)
    print(f"Audio file saved to {path_to_save}")
