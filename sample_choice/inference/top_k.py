from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")

#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#top_k - musi być int > 0
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_k1.wav", top_k=5.0)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_k2.wav", top_k=25)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_k3.wav", top_k=50)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_k4.wav", top_k=75)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_k5.wav", top_k=100)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_k6.wav", top_k=130)
""" Nie ma wpływu."""




