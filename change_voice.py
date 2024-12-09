from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager, SHORT_LATENT_KEY
from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler
from embedding_modifier.handlers.short_latent_to_short_latent_model_handler import ShortLatentToShortLatentModelHandler
from settings import SAMPLE_PATH_DB_KEY
from voice_modifier import VoiceModifier


psldb = ParametersShortLatentsDatabaseManager()
basic_path = "/app/results/change_voice_tests/"

input_speaker_sample_path = "/app/Resources/ready_audio_samples/common_voice_pl_20606767.wav"

def retrieve_sh_latent_and_parameters_from_path(path):
    record = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path)
    return record[SHORT_LATENT_KEY], record.to_dict()


new_parameters =  {'path': '/app/Resources/ready_audio_samples/common_voice_pl_20606767.wav', 
                      'f0': 160.1, 'male': 99.88, 'variance': 0.006599999964237213, 'skewness': 1.3087,
                      'kurtosis': 16.753, 'intensity': 71.4872, 'jitter': 0.0275, 'shimmer': 0.1291, 'hnr': 7.8215,
                      'zero_crossing_rate': 0.1952, 'spectral_centroid': 3241.54, 'spectral_bandwidth': 2839.1392,
                      'spectral_flatness': 0.049300000071525574, 'spectral_roll_off': 5951.286,
                      'tonnetz_fifth_x': -0.0053, 'tonnetz_fifth_y': 0.0361, 'tonnetz_minor_x': 0.019,
                      'tonnetz_minor_y': -0.0194, 'tonnetz_major_x': 0.0116, 'tonnetz_major_y': -0.0125,
                      'chroma_c': 0.5601000189781189, 'chroma_c_sharp': 0.5078999996185303,
                      'chroma_d': 0.4683000147342682, 'chroma_d_sharp': 0.46939998865127563,
                      'chroma_e': 0.4422000050544739, 'chroma_f': 0.515500009059906,
                      'chroma_f_sharp': 0.5041000247001648, 'chroma_g': 0.6205000281333923,
                      'chroma_g_sharp': 0.6198999881744385, 'chroma_a': 0.5618000030517578,
                      'chroma_a_sharp': 0.614300012588501, 'chroma_b': 0.5916000008583069,
                      'voiced_segments_per_second': 2.056962013244629, 'mean_voiced_segments_length': 0.26153847575187683,
                      'mean_unvoiced_segments_length': 0.21583333611488342, 'loudness_peaks_per_second': 3.7676610946655273,
                      'f0_fluctuations': 0.4184642434120178, 'f1': 664.5595703125, 'f1_bandwidth': 1096.8245849609375,
                      'f2': 1706.19677734375, 'f2_bandwidth': 760.7246704101562, 'f3': 2803.71923828125,
                      'f3_bandwidth': 697.6337890625}

# modifier = VoiceModifier(ShortLatentToShortLatentModelHandler, model_version="7")

# short_latent, _ = retrieve_sh_latent_and_parameters_from_path(input_speaker_sample_path)
# new_parameters['f0'] = 160
# modifier.generate_output(f"{basic_path}male_108_to_160_C.wav", new_parameters, short_latent=short_latent)
# new_parameters['f0'] = 108.1
# modifier.generate_output(f"{basic_path}male_108_to_108_C.wav", new_parameters, short_latent=short_latent)
# new_parameters['f0'] = 80
# modifier.generate_output(f"{basic_path}male_108_to_80_C.wav", new_parameters, short_latent=short_latent)

# modifier = VoiceModifier(ParametersToShortLatentModelHandler, model_version="5")
# short_latent, _ = retrieve_sh_latent_and_parameters_from_path(input_speaker_sample_path)
# new_parameters['f0'] = 160
# modifier.generate_output(f"{basic_path}male_108_to_160_A_12.wav", new_parameters)
# new_parameters['f0'] = 108.1
# modifier.generate_output(f"{basic_path}male_108_to_108_A_12.wav", new_parameters)
# new_parameters['f0'] = 80
# modifier.generate_output(f"{basic_path}male_108_to_80_A_12.wav", new_parameters)

# change CHOSEN_PARAMETERS_KEYS in embedding_modifier/models/model.py to: ALL_KEYS from parameters retriever
modifier = VoiceModifier(ParametersToShortLatentModelHandler, model_version="19")
short_latent, _ = retrieve_sh_latent_and_parameters_from_path(input_speaker_sample_path)
new_parameters['f0'] = 160
modifier.generate_output(f"{basic_path}male_108_to_160_A_44.wav", new_parameters)
new_parameters['f0'] = 108.1
modifier.generate_output(f"{basic_path}male_108_to_108_A_44.wav", new_parameters)
new_parameters['f0'] = 80
modifier.generate_output(f"{basic_path}male_108_to_80_A_44.wav", new_parameters)
