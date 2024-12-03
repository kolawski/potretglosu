import os

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

RESULTS_DIR = "/app/results"
SAMPLES_DIR = "/app/Resources/ready_audio_samples"
MODEL_PATH = "/image_resources/models/XTTS-v2"  # Tylko do katalogu
CONFIG_PATH = f"{MODEL_PATH}/config.json"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, use_deepspeed=False)
model.cuda()


print("Computing speaker latents...")
gpt_cond_latent_gruby, speaker_embedding_gruby = model.get_conditioning_latents(audio_path=[f"{SAMPLES_DIR}/common_voice_pl_40006748.wav"])

gpt_cond_latent_gruby = gpt_cond_latent_gruby.mean(dim=1, keepdim=True).repeat(1, 32, 1)
# gpt_cond_latent_gruby = gpt_cond_latent_gruby[:, :2, :].repeat(1, 32, 1)

# gpt_cond_latent_cienki, speaker_embedding_cienki = model.get_conditioning_latents(audio_path=[f"{SAMPLES_DIR}/refcienki.mp3"])

# mid_latent = torch.lerp(gpt_cond_latent_gruby, gpt_cond_latent_cienki, 0.5)
# mid_embedding = torch.lerp(speaker_embedding_gruby, speaker_embedding_cienki, 0.5)

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
# print(type(mid_latent))
# print(mid_latent.shape)
# print(type(mid_embedding))
# print(mid_embedding.shape)
print("Inference...")
out = model.inference(
    "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować",
    "pl",
    gpt_cond_latent_gruby,
    speaker_embedding_gruby,
    #temperature=0.7, # Add custom parameters here
)
torchaudio.save(f"{RESULTS_DIR}/speaker_3_repeat_mean_2.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

# spróbować zmienić model na najnowszą wersję
# sprawdzić która metoda łączenia najmniej traci na jakości
# trzeba się zorientować na co wpływa latent, a na co embedding
# model do testowania
