"""Module for retrieving latent and speaker embedding from audio sample and adding it into data base"""
import csv
from pathlib import Path

from tqdm import tqdm

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from settings import SAMPLES_DIRECTORY_PATH
from utils.xtts_handler import XTTSHandler, DEFAULT_PATH
from utils.metrics_retriever import retrieve_metrics


samples_dir = Path(SAMPLES_DIRECTORY_PATH)
report_path = "/app/results/other_tests/add_to_embedding_db_report.csv"
db_path = "/app/Resources/databases/paths_inference_embeddings_latents.parquet"

xtts_handler = XTTSHandler()
db_manager = EmbeddingDatabaseManager(db_path=db_path)

file_paths = [str(file) for file in samples_dir.glob('*') if file.is_file()]
progress_bar = tqdm(total=len(file_paths), desc="Processing")

csv_file = open(report_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['path', 'embeddings_mse', 'latents_mse', 'latents_ssim'])

for sample_path in file_paths:
    print(f"Processing {repr(sample_path)}")

    gpt_cond_latent, speaker_embedding = xtts_handler.compute_latents(sample_path)
    xtts_handler.inference(speaker_embedding, gpt_cond_latent)
    inference_latent, inference_embedding = xtts_handler.compute_latents(DEFAULT_PATH)


    embeddings_mse = retrieve_metrics(speaker_embedding, inference_embedding, get_ssim=False)
    latent_ssim, latents_mse = retrieve_metrics(gpt_cond_latent, inference_latent)
    print(f"Embeddings MSE: {embeddings_mse}, Latents MSE: {latents_mse}, Latents SSIM: {latent_ssim}")
    csv_writer.writerow([sample_path, embeddings_mse, latents_mse, latent_ssim])

    db_manager.add_data(embedding=inference_embedding, latent=inference_latent, sample_path=sample_path)
    progress_bar.update(1)

csv_file.close()

db_manager.repartition(10)
db_manager.save_to_parquet()

print(db_manager._dd.compute().head())
