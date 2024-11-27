import numpy as np

from utils.embedding_converter import flat_to_torch
from utils.parameters_extractor import ParametersExtractor
from utils.xtts_handler import XTTSHandler

class ParametersRetriever:
    def __init__(self):
        self.parameters_extractor = ParametersExtractor()
        self.xtts_handler = XTTSHandler()

    def retrieve_parameters(self, embedding, latent):
        path = self.xtts_handler.inference(embedding, latent)
        return self.parameters_extractor.extract_parameters(path)


def get_only_chosen_parameters(parameters, chosen_parameters):  # TODO maybe move to other module
    return {key: parameters[key] for key in chosen_parameters}


def get_only_chosen_parameters_ndarray(parameters, chosen_parameters):
    return np.array(list(get_only_chosen_parameters(parameters, chosen_parameters).values()))


def prepare_parameters(parameters, chosen_parameters):
    parameters = get_only_chosen_parameters_ndarray(parameters, chosen_parameters)
    # z konwersją na float tak jak wagi w modelu, unsqueeze żeby było [1, 12] zamiast [12]
    parameters = flat_to_torch(parameters, len(chosen_parameters)).float().unsqueeze(0)
    return parameters
