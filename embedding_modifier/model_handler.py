import numpy as np
import torch

from embedding_modifier.modifier_model import ModifierModel, CHOSEN_PARAMETERS_KEYS
from embedding_modifier.parameters_retriever import prepare_parameters
from settings import MODEL_CHECKPOINT_PATH, EMBEDDING_SHAPE, LATENT_SHAPE, DEVICE
from utils.embedding_converter import flat_to_torch
from utils.parameters_extractor import ParametersExtractor
from utils.xtts_handler import XTTSHandler

DEFAULT_PATH = "/app/results/result.wav"
DEFAULT_PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."


class ModelHandler:
    def __init__(self, model_path=MODEL_CHECKPOINT_PATH, device=DEVICE):
        self.modifier_model = ModifierModel()
        self.modifier_model.to(device)
        self.modifier_model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.xtts_handler = XTTSHandler()
        self.parameters_extractor = None

    def generate_output(self, input_embedding, input_latent, expected_parameters, path=DEFAULT_PATH, phrase=DEFAULT_PHRASE, print_output_parameters=False):

        if isinstance(input_embedding, np.ndarray):
            input_embedding = flat_to_torch(input_embedding, EMBEDDING_SHAPE)
        if isinstance(input_latent, np.ndarray):
            input_latent = flat_to_torch(input_latent, LATENT_SHAPE)

        if isinstance(expected_parameters, dict):
            expected_parameters = prepare_parameters(expected_parameters, CHOSEN_PARAMETERS_KEYS)

        modified_embedding, modified_latent = self.modifier_model(expected_parameters, input_embedding, input_latent)

        self.xtts_handler.inference(modified_embedding, modified_latent, phrase=phrase, path=path)
        print(f"Output saved to {path}")

        if print_output_parameters:
            if self.parameters_extractor is None:
                self.parameters_extractor = ParametersExtractor()
            output_parameters = self.parameters_extractor.extract_parameters(path)
            print(f"Output parameters: {output_parameters}")
