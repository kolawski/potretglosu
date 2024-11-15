"""Module for retrieving info from audio sample and adding it into data base"""
from pathlib import Path

from embedding_database_manager import EmbeddingDatabaseManager
from latents_retriever import LatentsRetriever
from utils.parameters_extractor import ParametersExtractor


samples_dir = Path('/app/Resources/ready_audio_samples')

file_paths = [str(file) for file in samples_dir.glob('*') if file.is_file()]

latents_retriever = LatentsRetriever()
db_manager = EmbeddingDatabaseManager()
extractor = ParametersExtractor()

for sample_path in file_paths:
    print(f"Processing {repr(sample_path)}")

    gpt_cond_latent, speaker_embedding = latents_retriever.compute_latents(sample_path)
    parameters = extractor.extract_parameters(sample_path)

    db_manager.add_data(embedding=speaker_embedding, latent=gpt_cond_latent,
                        sample_path=sample_path, parameters=parameters)

db_manager.save_to_parquet()

print(db_manager._dd.compute().head())
