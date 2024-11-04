import os

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

SAMPLES_DIR = "/app/Resources/audio_samples"
MODEL_PATH = "/image_resources/models/XTTS-v2"  # Tylko do katalogu
CONFIG_PATH = f"{MODEL_PATH}/config.json"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, use_deepspeed=False)
model.cuda()


print("Computing speaker latents...")
gpt_cond_latent_gruby, speaker_embedding_gruby = model.get_conditioning_latents(audio_path=[f"{SAMPLES_DIR}/refgruby.mp3"])
gpt_cond_latent_cienki, speaker_embedding_cienki = model.get_conditioning_latents(audio_path=[f"{SAMPLES_DIR}/refcienki.mp3"])

mid_latent = torch.lerp(gpt_cond_latent_gruby, gpt_cond_latent_cienki, 0.5)
mid_embedding = torch.lerp(speaker_embedding_gruby, speaker_embedding_cienki, 0.5)

# for i in range(10):
#     print(f"Interpolating {i/10}")
#     mid_latent = torch.rand(1, 32, 1024)
#     mid_embedding = torch.rand(1, 512, 1)
#     out = model.inference(
#         "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować",
#         "pl",
#         mid_latent,
#         speaker_embedding_cienki,
#         #temperature=0.7, # Add custom parameters here
#     )
#     torchaudio.save(f"results/xtts_latent_random_embedding_cienki_{i}.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

# Kmeans
# 1 faza - glosy reprezentatywne
print(type(mid_latent))
print(mid_latent.shape)
print(type(mid_embedding))
print(mid_embedding.shape)
print("Inference...")
out = model.inference(
    "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować",
    "pl",
    gpt_cond_latent_cienki,
    speaker_embedding_cienki,
    #temperature=0.7, # Add custom parameters here
)
torchaudio.save("xtts_latent_cienki_emb_gruby.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

# spróbować zmienić model na najnowszą wersję
# sprawdzić która metoda łączenia najmniej traci na jakości
# trzeba się zorientować na co wpływa latent, a na co embedding
# model do testowania
