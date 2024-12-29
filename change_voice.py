from voice_modifier import VoiceModifier
from utils.parameters_extractor import (
    F0_FLUCTUATIONS_KEY,
    F0_KEY,
    F1_KEY,
    F2_KEY,
    F3_KEY,
    GENDER_KEY,
    HNR_KEY,
    JITTER_KEY,
    MEAN_VOICED_SEGMENTS_LENGTH_KEY,
    SHIMMER_KEY,
    SKEWNESS_KEY,
    VOICED_SEGMENTS_PER_SECOND_KEY,
    ALL_KEYS)

modifier = VoiceModifier()

parameters_from_path = modifier.retrieve_parameters_from_path("/app/Resources/ready_audio_samples/common_voice_pl_40314763.wav")
# można też użyć funkcji retrieve_parameters_from_embedding_latent, ale wydajniej jest to zrobić ze ścieżki, bo wtedy pobiera parametry z
# bazy, a nie je oblicza

print(f"Parameters from path: {parameters_from_path}")

# {'path': '/app/Resources/ready_audio_samples/common_voice_pl_40314763.wav', 'f0': 231.42, 'male': 0.14,
#                    'variance': 0.023399999365210533, 'skewness': 0.1394, 'kurtosis': 6.5924, 'intensity': 75.9948,
#                    'jitter': 0.0283, 'shimmer': 0.1015, 'hnr': 12.6834, 'zero_crossing_rate': 0.2932,
#                    'spectral_centroid': 2378.0788, 'spectral_bandwidth': 2085.0307, 'spectral_flatness': 0.0649000033736229,
#                    'spectral_roll_off': 4390.5507, 'tonnetz_fifth_x': 0.0138, 'tonnetz_fifth_y': -0.004, 'tonnetz_minor_x': -0.0651,
#                    'tonnetz_minor_y': -0.0456, 'tonnetz_major_x': -0.0004, 'tonnetz_major_y': 0.0149, 'chroma_c': 0.4846000075340271,
#                    'chroma_c_sharp': 0.38449999690055847, 'chroma_d': 0.4108000099658966, 'chroma_d_sharp': 0.3977999985218048,
#                    'chroma_e': 0.4027999937534332, 'chroma_f': 0.4025999903678894, 'chroma_f_sharp': 0.3898000121116638,
#                    'chroma_g': 0.3968000113964081, 'chroma_g_sharp': 0.5189999938011169, 'chroma_a': 0.6820999979972839,
#                    'chroma_a_sharp': 0.695900022983551, 'chroma_b': 0.6085000038146973, 'voiced_segments_per_second': 2.1156558990478516,
#                    'mean_voiced_segments_length': 0.20266665518283844, 'mean_unvoiced_segments_length': 0.3100000023841858,
#                    'loudness_peaks_per_second': 4.341736793518066, 'f0_fluctuations': 0.1841025948524475, 'f1': 664.439697265625,
#                    'f1_bandwidth': 1085.020751953125, 'f2': 1821.8223876953125, 'f2_bandwidth': 831.6712646484375, 'f3': 2866.68310546875,
#                    'f3_bandwidth': 719.1000366210938}

parameters_from_path[F0_KEY] = 180.0

modifier.generate_sample_from_parameters(parameters_from_path,
                                         path="/app/tmp/voice_changer_tmp.wav",
                                         phrase="Lubię ciastka z kremem",
                                         # XTTS inference parameters
                                         temperature=0.7,
                                         length_penalty=1.0,
                                         repetition_penalty=10.0,
                                         top_k=50,
                                         top_p=0.85,
                                         do_sample=True,
                                         num_beams=1,
                                         speed=1.0,
                                         enable_text_splitting=False
                                         )
