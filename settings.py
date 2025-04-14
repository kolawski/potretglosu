DEFAULT_SR = 22050 # CURRENTLY NOT USED, BUT MAY BE USEFUL
EMBEDDINGS_DB = "/app/Resources/databases/paths_inference_embeddings_latents.parquet"
TSNE_DB = "/app/Resources/databases/tsne.parquet"
PARAMETERS_DB = "/app/Resources/databases/parameters.parquet"
PARAMETERS_SHORT_LATENTS_DB = "/app/Resources/databases/parameters_short_latents.parquet"
MODEL_PATH = "/image_resources/models/XTTS-v2"
MODEL_CONFIG_PATH = f"{MODEL_PATH}/config.json"
SAMPLES_DIRECTORY_PATH = "/app/Resources/ready_audio_samples"
TMP_DIRECTORY_PATH = "/app/tmp"
EMBEDDING_SHAPE = [1, 512, 1]
LATENT_SHAPE = [1, 32, 1024]
SHORT_LATENT_SHAPE = [1536]
LONG_LATENT_MODEL_CHECKPOINT_DIR = "/app/Resources/model_files/long_latent_model"
PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR = "/app/Resources/model_files/parameters_to_short_latent_model"
DIMENSION_LATENT_TO_LATENT_MODEL_CHECKPOINT_DIR = "/app/Resources/model_files/dimension_latent_to_latent_model"
SHORT_LATENT_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR = "/app/Resources/model_files/short_latent_to_short_latent_model"
DEVICE = 'cuda'
SAMPLE_PATH_DB_KEY = "path"
NORMALIZATION_DICT_PATH = "/app/Resources/configs/parameters_normalization_vector.npz"
NORMALIZATION_INFERENCE_PARAMS_DICT_PATH = "/app/Resources/configs/inference_parameters_normalization_vector.npz"

SHARED_DIRECTORY_PATH = "/app/shared"
VECTOR_SEARCHER_TEMP_WAV_PATH = f"{SHARED_DIRECTORY_PATH}/test.wav"

