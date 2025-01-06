from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")

#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#top_p - float >0 and <1
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p1.wav", top_p=0.01)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p2.wav", top_p=0.2)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p3.wav", top_p=0.4)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p4.wav", top_p=0.6)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p5.wav", top_p=0.8)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p6.wav", top_p=0.99)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p81.wav", top_p=0.01)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/top_p82.wav", top_p=0.01)

""" Wypowiedzi delikatnie się od siebie różnią (ale ma to miejce nawet dla tych samych top_p. Nie idzie to w żadnym kierunku
dla rosnącego p, zatem można uznać, że nie ma wpływu"""




