import random
import numpy
from utils.xtts_handler import XTTSHandler
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from pathlib import Path

db_manager = EmbeddingDatabaseManager()
xtts_handler = XTTSHandler()

embeddings = db_manager.get_all_values_from_column(EMBEDDING_KEY)
latents = db_manager.get_all_values_from_column(LATENT_KEY)

steps = 5
samples_per_step = 4

embedding_indices = numpy.zeros((steps,samples_per_step))
latent_indices = numpy.zeros((steps,samples_per_step))

for j in range(steps):
    for i in range(samples_per_step):
        random_indice = random.randint(0, len(embeddings) - 1)
        embedding_indices[j, i] = random_indice
        latent_indices[j, i] = random_indice

#embedding_indices = latent_indices = [random.randint(0, len(embeddings) - 1) for _ in range(4)]

for j in range(steps):
    folder_path = Path(f"/app/shared/krok{j}")
    folder_path.mkdir(parents=True, exist_ok=True)
    for i in range(samples_per_step):
        embedding = embeddings.iloc[int(embedding_indices[j, i])]
        latent = latents.iloc[int(latent_indices[j, i])]
        xtts_handler.inference(embedding, latent, f"/app/shared/krok{j}/test{i}.wav", " To jest napad! ")
