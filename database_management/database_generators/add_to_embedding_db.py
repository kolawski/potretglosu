"""Module for retrieving latent and speaker embedding from audio sample and adding it into data base"""
from pathlib import Path

from tqdm import tqdm

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from settings import SAMPLES_DIRECTORY_PATH
from utils.latents_retriever import LatentsRetriever


samples_dir = Path(SAMPLES_DIRECTORY_PATH)

latents_retriever = LatentsRetriever()
db_manager = EmbeddingDatabaseManager()

file_paths = [str(file) for file in samples_dir.glob('*') if file.is_file()]
progress_bar = tqdm(total=len(file_paths), desc="Processing")

for sample_path in file_paths:
    print(f"Processing {repr(sample_path)}")

    gpt_cond_latent, speaker_embedding = latents_retriever.compute_latents(sample_path)

    db_manager.add_data(embedding=speaker_embedding, latent=gpt_cond_latent, sample_path=sample_path)
    progress_bar.update(1)

db_manager.repartition(30)
db_manager.save_to_parquet()

print(db_manager._dd.compute().head())
