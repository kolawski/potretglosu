from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")
latent1, embedding1 = xtts.compute_latents("/app/shared/probka3.wav")

#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#enable_text_splitting - True/False
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting11.wav", do_sample=True)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting12.wav", do_sample=True)
# xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/enable_text_splitting13.wav", do_sample=True)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting21.wav", do_sample=False)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/enable_text_splitting22.wav", do_sample=False)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/enable_text_splitting23.wav", do_sample=False)
""" UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.85` 
-- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`."""