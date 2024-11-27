from pathlib import Path

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from utils.parameters_extractor import ParametersExtractor
from utils.xtts_handler import XTTSHandler
from settings import TMP_DIRECTORY_PATH


SAVE_PATH = "/app/results/histograms_same_embedding_random_latents/"

class ParametersComparator:
    def __init__(self, save_path, clean_tmp=False):
        # TODO docstrings
        self.save_path = save_path
        self.clean_tmp = clean_tmp
        self.edb = EmbeddingDatabaseManager()
        self.extractor = ParametersExtractor()
        self.xtts_model = XTTSHandler()

    def __del__(self):
        if self.clean_tmp:
            tmp_path = Path(TMP_DIRECTORY_PATH)
            for file in tmp_path.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

    def process_latent_and_embedding(self, embedding, latent, path, phrase):
        # Inference
        self.xtts_model.inference(embedding, latent, path, phrase)

        # Retrieve parameters
        return self.extractor.extract_parameters(path)
    
    @staticmethod
    def convert_list_of_dicts_to_dict_of_lists(list_of_dicts):
        dict_of_lists = {}
        for key in list_of_dicts[0].keys():
            dict_of_lists[key] = [d[key] for d in list_of_dicts]
        return dict_of_lists
