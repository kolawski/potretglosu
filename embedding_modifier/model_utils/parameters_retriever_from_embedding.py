from utils.parameters_extractor import ParametersExtractor
from utils.xtts_handler import XTTSHandler

class ParametersRetrieverFromEmbedding:
    def __init__(self):
        self.parameters_extractor = ParametersExtractor()
        self.xtts_handler = XTTSHandler()

    def retrieve_parameters(self, embedding, latent):
        path = self.xtts_handler.inference(embedding, latent)
        return self.parameters_extractor.extract_parameters(path)
