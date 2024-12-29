
import numpy as np
import torch

from embedding_modifier.handlers.model_handler import ModelHandler, DEFAULT_PATH, DEFAULT_PHRASE
from embedding_modifier.models.short_latent_to_short_latent_model import ShortLatentToShortLatentModel
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.handlers.dimension_latent_to_latent_model_handler import DimensionLatentToLatentModelHandler
from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import SHORT_LATENT_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, SHORT_LATENT_SHAPE, DEVICE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH
from utils.embedding_converter import flat_to_torch


class ShortLatentToShortLatentModelHandler(ModelHandler):
    def __init__(self, model_version="1", model_dir=SHORT_LATENT_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, device=DEVICE,
                 normalization_dict_path=NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, dimensions_latent_to_latent_model_version="1"):
        super().__init__(ShortLatentToShortLatentModel, model_version, model_dir, device=device)
        self.parameters_noramlization_dict = load_normalization_dict(normalization_dict_path)
        self.dimensions_latent_to_latent_model_version = dimensions_latent_to_latent_model_version
        self._dimension_latent_to_latent_model_handler = None

    @property
    def dimension_latent_to_latent_model_handler(self):
        if self._dimension_latent_to_latent_model_handler is None:
            self._dimension_latent_to_latent_model_handler = DimensionLatentToLatentModelHandler(self.dimensions_latent_to_latent_model_version)
        return self._dimension_latent_to_latent_model_handler

    def generate_output(self, short_latent, expected_parameters, path=DEFAULT_PATH, phrase=DEFAULT_PHRASE, print_output_parameters=False):

        recreated_short_latent = self.inference(short_latent, expected_parameters)

        self.dimension_latent_to_latent_model_handler.generate_output(recreated_short_latent,
                                                                       path=path,
                                                                       phrase=phrase,
                                                                       print_output_parameters=print_output_parameters)
        
        return recreated_short_latent

    def inference(self, short_latent, parameters):
        if isinstance(parameters, dict):
            parameters = prepare_parameters(parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)
        if isinstance(parameters, np.ndarray):
            parameters = flat_to_torch(parameters, (len(CHOSEN_PARAMETERS_KEYS),))
        if isinstance(short_latent, np.ndarray):
            short_latent = flat_to_torch(short_latent, SHORT_LATENT_SHAPE)

        short_latent_and_params = torch.cat((parameters.squeeze(), short_latent), dim=0)
        
        recreated_short_latent = self.modifier_model((short_latent_and_params,)).reshape(SHORT_LATENT_SHAPE) # tuple workaround for batch processing in model's forward

        return recreated_short_latent
