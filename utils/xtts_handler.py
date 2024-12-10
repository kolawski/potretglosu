import numpy as np
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from settings import DEFAULT_SR, TMP_DIRECTORY_PATH, MODEL_CONFIG_PATH, MODEL_PATH, EMBEDDING_SHAPE, LATENT_SHAPE
from utils.embedding_converter import flat_to_torch
from utils.exceptions import VoiceModifierError


DEFAULT_PATH = f"{TMP_DIRECTORY_PATH}/xtts_handler_tmp.wav"
DEFAULT_PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."

class XTTSHandler:
    # TODO docstrings
    def __init__(self, model_path=MODEL_PATH, config_path=MODEL_CONFIG_PATH):
        self.load_model(model_path, config_path)

    def compute_latents(self, *audio_paths):
        """Computes single speaker latents for all audio files specified in audio_paths

        :return: extracted gpt conditional latent and speaker embedding
        :rtype: tuple of torch.Tensor
        """
        for audio_path in audio_paths:
            if not Path(audio_path).is_file():
                raise VoiceModifierError(f"Audio file {audio_path} does not exist")
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
            audio_path=audio_path,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=True, # False by default
            load_sr=22050,)
        print("Successfully computed speaker latents")
        return gpt_cond_latent, speaker_embedding

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
        self.xtts_model = Xtts.init_from_config(config)
        self.xtts_model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
        self.xtts_model.cuda()
        print("Successfully loaded model")

    def inference(self, embedding, latent, path=DEFAULT_PATH,
                  phrase=DEFAULT_PHRASE, sr=DEFAULT_SR):
        if isinstance(embedding, np.ndarray):
            embedding = flat_to_torch(embedding, EMBEDDING_SHAPE)
        if isinstance(latent, np.ndarray):
            latent = flat_to_torch(latent, LATENT_SHAPE)
        print("Inference...")
        out = self.xtts_model.inference(
            phrase,
            "pl",
            latent,
            embedding,
            #temperature=0.7, # Add custom parameters here
        )
        torchaudio.save(path, torch.tensor(out["wav"]).unsqueeze(0), sr)

        return path
