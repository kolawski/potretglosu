from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")

#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#repetition penalty - float większy od 0, num_beams NIE jest potrzebna

# xtts.inference(embedding, latent, f"/app/shared/inference_tests/repetition_penalty4.wav", repetition_penalty=0.2, num_beams=5)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/repetition_penalty5.wav", repetition_penalty=1.0, num_beams=5)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/repetition_penalty6.wav", repetition_penalty=3.0, num_beams=5)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/repetition_penalty7.wav", repetition_penalty=5.0, num_beams=5)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/repetition_penalty8.wav", repetition_penalty=10.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/repetition_penalty50.wav", top_k=5.0)
"""WNIOSKI: Niższe wartości powodują brak jakichkolwiek słów lub ucięcie wypowiedzi. Już 5 jest ok, wyżej nie ma żadnych istotnych różnic."""

