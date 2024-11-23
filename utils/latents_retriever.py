from pathlib import Path

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from utils.exceptions import VoiceModifierError
from settings import MODEL_PATH, MODEL_CONFIG_PATH

class LatentsRetriever:
    def __init__(self, model=None):
        """Constructor

        :param model: model to be used for latents computing, defaults to None (model will be loaded from MODEL_PATH)
        :type model: TTS.tts.models.xtts.Xtta, optional
        """
        if model is None:
            self.load_model()
        else:
            self._model = model

    def load_model(self, model_path=MODEL_PATH, config_path=MODEL_CONFIG_PATH):
        """Loads model specified in settingsas a property

        :param model_path: path to a model directory, defaults to MODEL_PATH
        :type model_path: str, optional
        :param config_path: path to model's config, defaults to MODEL_CONFIG_PATH
        :type config_path: str, optional
        """
        if not Path(model_path).is_dir():
            raise VoiceModifierError(f"Model directory {model_path} does not exist")
        if not Path(config_path).is_file():
            raise VoiceModifierError(f"Model config file {config_path} does not exist")
        print("Loading model...")
        config = XttsConfig()
        config.load_json(config_path)
        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
        self._model.cuda()
        print("Successfully loaded model")

    def compute_latents(self, *audio_paths):
        """Computes single speaker latents for all audio files specified in audio_paths

        :return: extracted gpt conditional latent and speaker embedding
        :rtype: tuple of torch.Tensor
        """
        for audio_path in audio_paths:
            if not Path(audio_path).is_file():
                raise VoiceModifierError(f"Audio file {audio_path} does not exist")
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
            audio_path=audio_path,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=True, # False by default
            load_sr=22050,)
        print("Successfully computed speaker latents")
        return gpt_cond_latent, speaker_embedding
