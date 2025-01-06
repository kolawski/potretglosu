from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent1, embedding1 = xtts.compute_latents("/app/shared/speakers/speaker_1_30s.wav")
latent2, embedding2 = xtts.compute_latents("/app/shared/speakers/speaker_2_30s_women.wav")
latent3, embedding3 = xtts.compute_latents("/app/shared/speakers/speaker_3_30s.wav")
latent4, embedding4 = xtts.compute_latents("/app/shared/speakers/speaker_4_40s.wav")
latent5, embedding5 = xtts.compute_latents("/app/shared/speakers/speaker_5_20s.wav")

