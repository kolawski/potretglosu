
import numpy as np

from embedding_modifier.handlers.model_handler import ModelHandler, DEFAULT_PATH, DEFAULT_PHRASE
from embedding_modifier.models.parameters_to_short_latent_model import ParametersToShortLatentModel
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, EMBEDDING_SHAPE, LATENT_SHAPE, DEVICE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH
from utils.embedding_converter import flat_to_torch, torch_to_flat
from utils.parameters_extractor import ParametersExtractor



class ParametersToShortLatentModelHandler(ModelHandler):
    def __init__(self, model_dir=PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, device=DEVICE,
                 normalization_dict_path=NORMALIZATION_INFERENCE_PARAMS_DICT_PATH):
        super().__init__(ParametersToShortLatentModel, model_dir, normalization_dict_path, device=device)
        self.parameters_extractor = None

    def generate_output(self, expected_parameters, path=DEFAULT_PATH, phrase=DEFAULT_PHRASE, print_output_parameters=False):

        if isinstance(expected_parameters, dict):
            expected_parameters = prepare_parameters(expected_parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)

        short_latent = torch_to_flat(self.modifier_model(expected_parameters))
        latent_mean_dimension = short_latent[:1024]
        embedding = short_latent[-512:]

        modified_embedding = flat_to_torch(embedding, EMBEDDING_SHAPE)
        recreated_latent = np.tile(latent_mean_dimension, (32, 1))
        modified_latent = flat_to_torch(recreated_latent, LATENT_SHAPE)

        self.xtts_handler.inference(modified_embedding, modified_latent, phrase=phrase, path=path)
        print(f"Output saved to {path}")

        if print_output_parameters:
            if self.parameters_extractor is None:
                self.parameters_extractor = ParametersExtractor()
            output_parameters = self.parameters_extractor.extract_parameters(path)
            print(f"Output parameters: {output_parameters}")
