from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")
latent1, embedding1 = xtts.compute_latents("app/Resources/ready_audio_samples/common_voice_pl_20547774.wav")
latent2, embedding2 = xtts.compute_latents("app/Resources/ready_audio_samples/common_voice_pl_20554619.wav")
latent3, embedding3 = xtts.compute_latents("app/Resources/ready_audio_samples/common_voice_pl_20555528.wav")
#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#length_penalty - dowolny float, potrzebny jest num_beams
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty1.wav", length_penalty=-5.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty2.wav", length_penalty=-3.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty3.wav", length_penalty=-1.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty4.wav", length_penalty=0.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty5.wav", length_penalty=1.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty6.wav", length_penalty=3.0, num_beams=5)
xtts.inference(embedding, latent, f"/app/shared/inference_tests/length_penalty7.wav", length_penalty=5.0, num_beams=5)
""" WNIOSKI: nie ma żadnych istotnych różnic (jeśli są minimalne, to po prostu wpływa na nie fakt generacji).
Ostatnia próbka zawierała 25 s ciszy, ale to może być przypadek (nawet jeśli nie, to nie jest tu istotne), 
a poza tym nie ma nic do samej wypowiedzi."""




