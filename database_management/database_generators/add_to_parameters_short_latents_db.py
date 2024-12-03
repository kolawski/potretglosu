"""Module for retrieving parameters from audio sample and adding it into data base"""
import gc
import time

import numpy as np
from tqdm import tqdm

from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from settings import SAMPLE_PATH_DB_KEY
from utils.embedding_converter import numpy_embedding_latent_to_short
from utils.parameters_extractor import ParametersExtractor
from utils.xtts_handler import XTTSHandler, DEFAULT_PATH


parameters_extractor = ParametersExtractor()
psldb = ParametersShortLatentsDatabaseManager()
edb = EmbeddingDatabaseManager()
xtts = XTTSHandler()
progress_bar = tqdm(total=2400, desc="Processing")

embeddings_dd = edb.dd
for i, (_, series) in enumerate(embeddings_dd.iterrows()):
    path = series[SAMPLE_PATH_DB_KEY]
    embedding = series[EMBEDDING_KEY]
    latent = series[LATENT_KEY]

    short_latent = numpy_embedding_latent_to_short(embedding, latent)
    xtts.inference(embedding, latent)
    parameters = parameters_extractor.extract_parameters(DEFAULT_PATH)

    progress_bar.update(1)

    nan_occured = False
    for key, value in parameters.items():
        if not isinstance(value, (int, float, np.float32, np.float64)) or np.isnan(value):
            print(f"Value for path {path} for key '{key}' is not a number: {value}")
            nan_occured = True
            break

    if nan_occured:
        continue

    psldb.add_data(path, short_latent, parameters)

del parameters_extractor
del edb
del xtts
gc.collect()

time.sleep(60)

psldb.repartition(30)
psldb.save_to_parquet()

print(psldb._dd.compute().head())
