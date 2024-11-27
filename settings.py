DEFAULT_SR = 22050 # to be adjusted, when audio samples are provided CURRENTLY NOT USED, BUT MAY BE USEFUL
EMBEDDINGS_DB = "/app/Resources/databases/paths_embeddings_latents.parquet"
TSNE_DB = "/app/Resources/databases/tsne.parquet"
PARAMETERS_DB = "/app/Resources/databases/parameters.parquet"
MODEL_PATH = "/image_resources/models/XTTS-v2"
MODEL_CONFIG_PATH = f"{MODEL_PATH}/config.json"
SAMPLES_DIRECTORY_PATH = "/app/Resources/ready_audio_samples"
TMP_DIRECTORY_PATH = "/app/tmp"
EMBEDDING_SHAPE = [1, 512, 1]
LATENT_SHAPE = [1, 32, 1024]
MODEL_CHECKPOINT_PATH = "/app/Resources/model_files/checkpoint.pth"
DEVICE = 'cuda'
SAMPLE_PATH_DB_KEY = "path"

# chosen parameters are set in embeding_modifier/modifier_model.py
