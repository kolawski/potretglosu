from utils.xtts_handler import XTTSHandler

xtts = XTTSHandler()

latent, embedding = xtts.compute_latents("/app/shared/sample_3.wav")
latent1, embedding1 = xtts.compute_latents("/app/Resources/ready_audio_samples/common_voice_pl_20547774.wav")
latent2, embedding2 = xtts.compute_latents("/app/Resources/ready_audio_samples/common_voice_pl_20554619.wav")
latent3, embedding3 = xtts.compute_latents("/app/Resources/ready_audio_samples/common_voice_pl_20555528.wav")
#teraz jest ustawiona defaultowa wypowiedź w ustawieniach

#length_penalty - dowolny float, potrzebny jest num_beams
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans1.wav", num_beams=0)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans2.wav", num_beams=2.0)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans3.wav", num_beams=-3)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans4.wav", num_beams=4)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans5.wav", num_beams=5)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans6.wav", num_beams=6)
xtts.inference(embedding1, latent1, f"/app/shared/inference_tests/num_beans7.wav", num_beams=7)
""" WNIOSKI: nie ma wpływu"""
"""ValueError: `num_beams` has to be an integer strictly 
greater than 1, but is 0. For `num_beams` == 1, one should make use of `greedy_search` instead."""
