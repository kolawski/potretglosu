from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")

#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#enable_text_splitting - True/False
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting11.wav", enable_text_splitting=True)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting12.wav", enable_text_splitting=True)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting13.wav", enable_text_splitting=True)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting21.wav", enable_text_splitting=False)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting22.wav", enable_text_splitting=False)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting23.wav", enable_text_splitting=False)
""" WNIOSKI: Wypowiedzi są praktycznie identyczne, nie ma wpływu."""