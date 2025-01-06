from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/inference_parameters_test_speed_1_0.wav")
latent1, embedding1 = xtts.compute_latents("/app/shared/probka1.wav")
latent2, embedding2 = xtts.compute_latents("/app/shared/probka2.wav")
latent3, embedding3 = xtts.compute_latents("/app/shared/probka3.wav")
#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#temperature - musi być > 0 i float, otherwise next token scores will be invalid
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_1.wav", temperature=0.01)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_2.wav", temperature=0.2)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_3.wav", temperature=0.5)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_4.wav", temperature=1.0)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_5.wav", temperature=3.0)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_6.wav", temperature=5.0)
# xtts.inference(embedding, latent, f"/app/shared/inference_tests/temperature2_7.wav", temperature=8.0)

xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_1.wav", temperature=0.01)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_2.wav", temperature=0.2)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_3.wav", temperature=0.5)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_4.wav", temperature=1.0)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_5.wav", temperature=3.0)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_6.wav", temperature=5.0)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/temperature5_7.wav", temperature=8.0)

xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_1.wav", temperature=0.01)
xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_2.wav", temperature=0.2)
xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_3.wav", temperature=0.5)
xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_4.wav", temperature=1.0)
xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_5.wav", temperature=3.0)
xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_6.wav", temperature=5.0)
xtts.inference(embedding2, latent2, f"/app/shared/inference_tests/temperature3_7.wav", temperature=8.0)

xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_1.wav", temperature=0.01)
xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_2.wav", temperature=0.2)
xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_3.wav", temperature=0.5)
xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_4.wav", temperature=1.0)
xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_5.wav", temperature=3.0)
xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_6.wav", temperature=5.0)
xtts.inference(embedding3, latent3, f"/app/shared/inference_tests/temperature4_7.wav", temperature=8.0)
""" WNIOSKI: temperature=5.0 rzeczywiście jest bardziej emocjonalna w sensie (jakby się miał zaraz rozpłakać),
temperature=100.0 nie brzmi dużo bardziej emocjonalnie, a liczy się bardzo długo, zatem istnieje jakieś optimum.
Na stronie XTTS też jest przykład dla temperature 0.7 z wprowadzonym testem, więc parametr ten faktycznie musi mieć znaczenie.
Natomiast nie działa to na tyle dobrze, by wyniki były powtarzalne na wszystkich próbkach"""
