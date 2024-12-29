# # Skrypt testowy Kuby do testowania różnych innych skryptów

# import csv
# from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
# from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
# from utils.embedding_converter import flat_to_torch
# from utils.metrics_retriever import retrieve_metrics
# from utils.parameters_extractor import GENDER_KEY
# from utils.xtts_handler import XTTSHandler, DEFAULT_PATH
# from settings import SAMPLE_PATH_DB_KEY, LATENT_SHAPE

# edb = EmbeddingDatabaseManager()
# # psldb = ParametersShortLatentsDatabaseManager()
# # pomiędzy bazami danych trzeba przechodzić po ścieżkach - niestety trzeba przechowywać te ścieżki razem z embeddingami/latentami przy liczeniu
# # albo w słownikach, albo w osobnych obiektach utworzonej do tego celu dataclass
# csv_file = open('/app/results/other_tests/different latents.csv', mode='a', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['latents mse', 'latents ssim'])

# # xtts = XTTSHandler()
# random_record = edb.get_random_record()
# # embedding = random_record[EMBEDDING_KEY]
# latent_1 = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)

# for i in range(60):
#     random_record = edb.get_random_record()
#     path = random_record[SAMPLE_PATH_DB_KEY]
#     print(f"random path: {path}")
#     latent_2 = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)

#     # xtts.inference(embedding, latent)
#     # new_latent, new_embedding = xtts.compute_latents(DEFAULT_PATH)

#     latent_ssim, latent_mse = retrieve_metrics(latent_1, latent_2, get_ssim=True, get_m2e=True)
#     # embedding_mse = retrieve_metrics(embedding, new_embedding, get_ssim=False, get_m2e=True)

#     csv_writer.writerow([latent_mse, latent_ssim])

# csv_file.close()


# Skrypt testowy Kuby do testowania różnych innych skryptów

# import csv
# from embedding_modifier.model_utils.common import load_normalization_dict
# from embedding_modifier.model_utils.parameters_utils import prepare_parameters
# from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
# from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
# from utils.embedding_converter import flat_to_torch
# from utils.metrics_retriever import retrieve_metrics
# from utils.parameters_extractor import GENDER_KEY, ALL_KEYS
# from utils.xtts_handler import XTTSHandler, DEFAULT_PATH
# from settings import SAMPLE_PATH_DB_KEY, LATENT_SHAPE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH

# edb = EmbeddingDatabaseManager()
# psldb = ParametersShortLatentsDatabaseManager()
# # pomiędzy bazami danych trzeba przechodzić po ścieżkach - niestety trzeba przechowywać te ścieżki razem z embeddingami/latentami przy liczeniu
# # albo w słownikach, albo w osobnych obiektach utworzonej do tego celu dataclass
# csv_file = open('/app/results/other_tests/does_embedding_and_latent_mse_correlate_with_parameters_mse.csv', mode='a', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['embedding mse', 'latents mse', 'latents ssim', 'parameters_mse', 'gender'])

# # csv_file_2 = open('/app/results/other_tests/does_embedding_and_latent_mse_correlate_with_gender.csv', mode='a', newline='')
# # csv_writer_2 = csv.writer(csv_file_2)
# # csv_writer_2.writerow(['embedding mse', 'latents mse', 'latents ssim', ])

# # xtts = XTTSHandler()
# # random_record = edb.get_random_record()
# # embedding = random_record[EMBEDDING_KEY]
# # latent = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)

# for i in range(500):
#     normalization_dict = load_normalization_dict(NORMALIZATION_INFERENCE_PARAMS_DICT_PATH)

#     random_record = edb.get_random_record()
#     path_1 = random_record[SAMPLE_PATH_DB_KEY]
#     print(f"random path: {path_1}")
#     latent_1 = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)
#     embedding_1 = random_record[EMBEDDING_KEY]
#     psl_record_1 = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path_1)
#     gender_1 = psl_record_1[GENDER_KEY]
#     parameters_1 = prepare_parameters(psl_record_1, ALL_KEYS, normalization_dict)
#     print(f"parameters_1: {parameters_1}")
#     print(f"gender_1: {gender_1}")

#     random_record_2 = edb.get_random_record()
#     path_2 = random_record_2[SAMPLE_PATH_DB_KEY]
#     latent_2 = flat_to_torch(random_record_2[LATENT_KEY], LATENT_SHAPE)
#     embedding_2 = random_record_2[EMBEDDING_KEY]
#     psl_record_2 = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path_2)
#     gender_2 = psl_record_2[GENDER_KEY]
#     parameters_2 = prepare_parameters(psl_record_2, ALL_KEYS, normalization_dict)
#     print(f"parameters_2: {parameters_2}")
#     print(f"gender_2: {gender_2}")


#     latent_ssim, latent_mse = retrieve_metrics(latent_1, latent_2, get_ssim=True, get_m2e=True)
#     embedding_mse = retrieve_metrics(embedding_1, embedding_2, get_ssim=False, get_m2e=True)
#     parmaeters_mse = retrieve_metrics(parameters_1, parameters_2, get_ssim=False, get_m2e=True)

#     gender_difference = abs(gender_1 - gender_2)
#     csv_writer.writerow([embedding_mse, latent_mse, latent_ssim, parmaeters_mse, gender_difference])

# csv_file.close()

# import csv
# from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
# from embedding_modifier.model_utils.common import load_normalization_dict
# from embedding_modifier.model_utils.parameters_utils import prepare_parameters
# from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
# from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
# from utils.embedding_converter import flat_to_torch
# from utils.metrics_retriever import retrieve_metrics
# from utils.parameters_extractor import ParametersExtractor, GENDER_KEY, ALL_KEYS
# from utils.xtts_handler import XTTSHandler, DEFAULT_PATH
# from settings import SAMPLE_PATH_DB_KEY, LATENT_SHAPE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH

# # edb = EmbeddingDatabaseManager()
# psldb = ParametersShortLatentsDatabaseManager()
# # pomiędzy bazami danych trzeba przechodzić po ścieżkach - niestety trzeba przechowywać te ścieżki razem z embeddingami/latentami przy liczeniu
# # albo w słownikach, albo w osobnych obiektach utworzonej do tego celu dataclass
# csv_file = open('/app/results/other_tests/12_parameters_random_mse.csv', mode='a', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['parameters_mse'])

# # csv_file_2 = open('/app/results/other_tests/does_embedding_and_latent_mse_correlate_with_gender.csv', mode='a', newline='')
# # csv_writer_2 = csv.writer(csv_file_2)
# # csv_writer_2.writerow(['embedding mse', 'latents mse', 'latents ssim', ])

# # xtts = XTTSHandler()
# # random_record = edb.get_random_record()
# # embedding = random_record[EMBEDDING_KEY]
# # latent = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)

# for i in range(500):
#     normalization_dict = load_normalization_dict(NORMALIZATION_INFERENCE_PARAMS_DICT_PATH)

#     random_record = psldb.get_random_record()
#     # path_1 = random_record[SAMPLE_PATH_DB_KEY]
#     # print(f"random path: {path_1}")
#     # latent_1 = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)
#     # embedding_1 = random_record[EMBEDDING_KEY]
#     # psl_record_1 = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path_1)
#     # gender_1 = psl_record_1[GENDER_KEY]
#     parameters_1 = prepare_parameters(random_record, CHOSEN_PARAMETERS_KEYS, normalization_dict)
#     print(f"parameters_1: {parameters_1}")
#     # print(f"gender_1: {gender_1}")

#     random_record_2 = psldb.get_random_record()
#     # path_2 = random_record_2[SAMPLE_PATH_DB_KEY]
#     # latent_2 = flat_to_torch(random_record_2[LATENT_KEY], LATENT_SHAPE)
#     # embedding_2 = random_record_2[EMBEDDING_KEY]
#     # psl_record_2 = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path_2)
#     # gender_2 = psl_record_2[GENDER_KEY]
#     parameters_2 = prepare_parameters(random_record_2, CHOSEN_PARAMETERS_KEYS, normalization_dict)
#     print(f"parameters_2: {parameters_2}")
#     # print(f"gender_2: {gender_2}")


#     # latent_ssim, latent_mse = retrieve_metrics(latent_1, latent_2, get_ssim=True, get_m2e=True)
#     # embedding_mse = retrieve_metrics(embedding_1, embedding_2, get_ssim=False, get_m2e=True)
#     parmaeters_mse = retrieve_metrics(parameters_1, parameters_2, get_ssim=False, get_m2e=True)

#     # gender_difference = abs(gender_1 - gender_2)
#     csv_writer.writerow([parmaeters_mse])
# csv_file.close()

# edb = EmbeddingDatabaseManager()
# psldb = ParametersShortLatentsDatabaseManager()
# extractor = ParametersExtractor()
# # pomiędzy bazami danych trzeba przechodzić po ścieżkach - niestety trzeba przechowywać te ścieżki razem z embeddingami/latentami przy liczeniu
# # albo w słownikach, albo w osobnych obiektach utworzonej do tego celu dataclass
# csv_file = open('/app/results/other_tests/parameters_mse_on_2_inferences_with_same_embedding_and_latent_2.csv', mode='a', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['parameters_12_mse', 'parameters_44_mse'])

# # csv_file_2 = open('/app/results/other_tests/does_embedding_and_latent_mse_correlate_with_gender.csv', mode='a', newline='')
# # csv_writer_2 = csv.writer(csv_file_2)
# # csv_writer_2.writerow(['embedding mse', 'latents mse', 'latents ssim', ])

# xtts = XTTSHandler()
# # random_record = edb.get_random_record()
# # embedding = random_record[EMBEDDING_KEY]
# # latent = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)
# normalization_dict = load_normalization_dict(NORMALIZATION_INFERENCE_PARAMS_DICT_PATH)

# for i in range(100):
#     edb_record = edb.get_random_record()
#     path = edb_record[SAMPLE_PATH_DB_KEY]
#     embedding = edb_record[EMBEDDING_KEY]
#     latent = edb_record[LATENT_KEY]

#     xtts.inference(embedding, latent, path=DEFAULT_PATH)
#     new_parameters_1 = extractor.extract_parameters(DEFAULT_PATH)

#     parameters_12_1 = prepare_parameters(new_parameters_1, CHOSEN_PARAMETERS_KEYS, normalization_dict)
#     parameters_44_1 = prepare_parameters(new_parameters_1, ALL_KEYS, normalization_dict)

#     xtts.inference(embedding, latent, path=DEFAULT_PATH)
#     new_parameters_2 = extractor.extract_parameters(DEFAULT_PATH)

#     parameters_12_2 = prepare_parameters(new_parameters_2, CHOSEN_PARAMETERS_KEYS, normalization_dict)
#     parameters_44_2 = prepare_parameters(new_parameters_2, ALL_KEYS, normalization_dict)

#     parameters_12_mse = retrieve_metrics(parameters_12_1, parameters_12_2, get_ssim=False)
#     parameters_44_mse = retrieve_metrics(parameters_44_1, parameters_44_2, get_ssim=False)

#     csv_writer.writerow([parameters_12_mse, parameters_44_mse])
# csv_file.close()

from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler
from embedding_modifier.handlers.long_latent_model_handler import LongLatentModelHandler
from embedding_modifier.handlers.short_latent_to_short_latent_model_handler import ShortLatentToShortLatentModelHandler
from embedding_modifier.handlers.dimension_latent_to_latent_model_handler import DimensionLatentToLatentModelHandler

handler = ParametersToShortLatentModelHandler(model_version="11")
handler.convert_to_onnx()
