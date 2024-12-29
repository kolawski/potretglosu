import csv

from tqdm import tqdm

from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler
from embedding_modifier.handlers.dimension_latent_to_latent_model_handler import DimensionLatentToLatentModelHandler
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, SAMPLE_PATH_DB_KEY, EMBEDDING_SHAPE, LATENT_SHAPE
from utils.embedding_converter import flat_to_torch
from utils.metrics_retriever import retrieve_metrics


wav_temp_path = "/app/results/models_efficiency_tests/random_model_12_inference.wav"
csv_file_path = "/app/results/models_efficiency_tests/model_19_embeddings_to_params_to_embeddings_deformation_test.csv"

iterations = 100

progress_bar = tqdm(total=iterations, desc="Processing")

file = open(csv_file_path, mode='a', newline='')
writer = csv.writer(file)
writer.writerow(["embeddings_mse", "latents_mse", "latents_ssim"])

normalization_dict = load_normalization_dict(NORMALIZATION_INFERENCE_PARAMS_DICT_PATH)

psldb = ParametersShortLatentsDatabaseManager()
edb = EmbeddingDatabaseManager()
handler_A = ParametersToShortLatentModelHandler(model_version="19")
handler_B = DimensionLatentToLatentModelHandler()

for i in range(iterations):
    psldb_record = psldb.get_random_record()
    parameters = prepare_parameters(psldb_record, CHOSEN_PARAMETERS_KEYS, normalization_dict)
    path = psldb_record[SAMPLE_PATH_DB_KEY]

    edb_record = edb.get_record_by_key(SAMPLE_PATH_DB_KEY, path)
    original_embedding = flat_to_torch(edb_record[EMBEDDING_KEY], EMBEDDING_SHAPE)
    original_latent = flat_to_torch(edb_record[LATENT_KEY], LATENT_SHAPE)

    short_latent = handler_A.inference(parameters)
    recreated_latent, recreated_embedding = handler_B.inference(short_latent, enforce_tensor_output=True)

    print(f"Shape of recreated latent: {recreated_latent.shape}")
    print(f"Shape of original latent: {original_latent.shape}")
    print(f"Shape of recreated embedding: {recreated_embedding.shape}")
    print(f"Shape of original embedding: {original_embedding.shape}")

    embeddings_mse = retrieve_metrics(original_embedding, recreated_embedding, get_ssim=False)
    latents_ssim, latents_mse = retrieve_metrics(original_latent, recreated_latent)

    writer.writerow([embeddings_mse, latents_mse, latents_ssim])

    progress_bar.update(1)
