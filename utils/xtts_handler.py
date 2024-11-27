import numpy as np
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from settings import DEFAULT_SR, TMP_DIRECTORY_PATH, MODEL_CONFIG_PATH, MODEL_PATH, EMBEDDING_SHAPE, LATENT_SHAPE
from utils.embedding_converter import flat_to_torch


DEFAULT_PATH = f"{TMP_DIRECTORY_PATH}/xtts_handler_tmp.wav"
DEFAULT_PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."

class XTTSHandler:
    # TODO docstrings
    def __init__(self):
        print("Loading model...")
        config = XttsConfig()
        config.load_json(MODEL_CONFIG_PATH)
        self.xtts_model = Xtts.init_from_config(config)
        self.xtts_model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, use_deepspeed=False)
        self.xtts_model.cuda()

    def inference(self, embedding, latent, path=DEFAULT_PATH,
                  phrase=DEFAULT_PHRASE, sr=DEFAULT_SR):
        print(f"Embedding shape: {embedding.shape}, latent shape: {latent.shape}") # REM
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
