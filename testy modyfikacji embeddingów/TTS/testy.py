import os

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

script_dir = os.path.dirname(__file__)
MODEL_PATH = fr"{script_dir}\pretrenowane modele\tts_models--multilingual--multi-dataset--xtts"  # Tylko do katalogu
CONFIG_PATH = fr"{script_dir}\pretrenowane modele\tts_models--multilingual--multi-dataset--xtts\config.json"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, use_deepspeed=False)
model.cuda()


print("Computing speaker latents...")
gpt_cond_latent_gruby, speaker_embedding_gruby = model.get_conditioning_latents(audio_path=["refgruby.mp3"])
gpt_cond_latent_cienki, speaker_embedding_cienki = model.get_conditioning_latents(audio_path=["refcienki.mp3"])

mid_latent = torch.lerp(gpt_cond_latent_gruby, gpt_cond_latent_cienki, 0.5)
mid_embedding = torch.lerp(speaker_embedding_gruby, speaker_embedding_cienki, 0.5)

print("Inference...")
out = model.inference(
    "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować",
    "pl",
    mid_latent,
    speaker_embedding_gruby,
    #temperature=0.7, # Add custom parameters here
)
torchaudio.save("xtts_latent_both_embedding_gruby.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

# spróbować zmienić model na najnowszą wersję
# sprawdzić która metoda łączenia najmniej traci na jakości
# trzeba się zorientować na co wpływa latent, a na co embedding
# model do testowania
