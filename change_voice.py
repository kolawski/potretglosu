from embedding_modifier.handlers.long_latent_model_handler import LongLatentModelHandler
from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from settings import SAMPLE_PATH_DB_KEY
from utils.parameters_extractor import (
    ParametersExtractor,
    F0_KEY,
    GENDER_KEY,
    SKEWNESS_KEY,
    JITTER_KEY,
    SHIMMER_KEY,
    HNR_KEY,
    VOICED_SEGMENTS_PER_SECOND_KEY,
    MEAN_VOICED_SEGMENTS_LENGTH_KEY,
    F0_FLUCTUATIONS_KEY,
    F1_KEY,
    F2_KEY,
    F3_KEY
)


model_handler = ParametersToShortLatentModelHandler()
# model_handler = LongLatentModelHandler()
edb = EmbeddingDatabaseManager()
pdb = ParametersDatabaseManager()
# extractor = ParametersExtractor()

input_speaker_sample_path = "/app/Resources/ready_audio_samples/common_voice_pl_20606767.wav"

# Original parameters: {'path': '/app/Resources/ready_audio_samples/common_voice_pl_20606767.wav', 
#                       'f0': 108.1, 'male': 99.88, 'variance': 0.006599999964237213, 'skewness': 1.3087,
#                       'kurtosis': 16.753, 'intensity': 71.4872, 'jitter': 0.0275, 'shimmer': 0.1291, 'hnr': 7.8215,
#                       'zero_crossing_rate': 0.1952, 'spectral_centroid': 3241.54, 'spectral_bandwidth': 2839.1392,
#                       'spectral_flatness': 0.049300000071525574, 'spectral_roll_off': 5951.286,
#                       'tonnetz_fifth_x': -0.0053, 'tonnetz_fifth_y': 0.0361, 'tonnetz_minor_x': 0.019,
#                       'tonnetz_minor_y': -0.0194, 'tonnetz_major_x': 0.0116, 'tonnetz_major_y': -0.0125,
#                       'chroma_c': 0.5601000189781189, 'chroma_c_sharp': 0.5078999996185303,
#                       'chroma_d': 0.4683000147342682, 'chroma_d_sharp': 0.46939998865127563,
#                       'chroma_e': 0.4422000050544739, 'chroma_f': 0.515500009059906,
#                       'chroma_f_sharp': 0.5041000247001648, 'chroma_g': 0.6205000281333923,
#                       'chroma_g_sharp': 0.6198999881744385, 'chroma_a': 0.5618000030517578,
#                       'chroma_a_sharp': 0.614300012588501, 'chroma_b': 0.5916000008583069,
#                       'voiced_segments_per_second': 2.056962013244629, 'mean_voiced_segments_length': 0.26153847575187683,
#                       'mean_unvoiced_segments_length': 0.21583333611488342, 'loudness_peaks_per_second': 3.7676610946655273,
#                       'f0_fluctuations': 0.4184642434120178, 'f1': 664.5595703125, 'f1_bandwidth': 1096.8245849609375,
#                       'f2': 1706.19677734375, 'f2_bandwidth': 760.7246704101562, 'f3': 2803.71923828125,
#                       'f3_bandwidth': 697.6337890625}

edb_record = edb.get_record_by_key(SAMPLE_PATH_DB_KEY, input_speaker_sample_path)
embedding = edb_record[EMBEDDING_KEY]
latent = edb_record[LATENT_KEY]

print(min(embedding), max(embedding))
print(min(latent), max(latent))


# Only parameters from embedding_modifier.modifier_model.py's CHOSEN_PARAMETERS_KEYS
# will be concidered. All of them must be specified.
expected_parameters = {
    F0_KEY: 180.0,
    GENDER_KEY: 0.8,
    SKEWNESS_KEY: 1.3087,
    JITTER_KEY: 0.0275,
    SHIMMER_KEY: 0.1291,
    HNR_KEY: 7.8215,
    VOICED_SEGMENTS_PER_SECOND_KEY: 2.05696,
    MEAN_VOICED_SEGMENTS_LENGTH_KEY: 0.26153,
    F0_FLUCTUATIONS_KEY: 0.41846,
    F1_KEY: 864.55,
    F2_KEY: 2706.1,
    F3_KEY: 3203.719,
    }

# model_handler.generate_output(embedding, latent, expected_parameters, print_output_parameters=True)
model_handler.generate_output(expected_parameters, print_output_parameters=True)

# original_parameters = pdb.get_record_by_key(SAMPLE_PATH_DB_KEY, input_speaker_sample_path).to_dict()
# new_parameters = extractor.extract_parameters(input_speaker_sample_path)

# print(f"Original parameters: {original_parameters}\n")
# print(f"Expected parameters: {expected_parameters}\n")
# print(f"New parameters: {new_parameters}")
