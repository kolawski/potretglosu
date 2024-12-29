
import numpy as np

from embedding_modifier.handlers.model_handler import ModelHandler, DEFAULT_PATH, DEFAULT_PHRASE
from embedding_modifier.models.long_latent_model import LongLatentModifierModel
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import LONG_LATENT_MODEL_CHECKPOINT_DIR, EMBEDDING_SHAPE, LATENT_SHAPE, DEVICE, NORMALIZATION_DICT_PATH
from utils.embedding_converter import flat_to_torch
from utils.parameters_extractor import ParametersExtractor



class LongLatentModelHandler(ModelHandler):
    def __init__(self, model_version="1", model_dir=LONG_LATENT_MODEL_CHECKPOINT_DIR, device=DEVICE,
                 normalization_dict_path=NORMALIZATION_DICT_PATH):
        super().__init__(LongLatentModifierModel, model_version, model_dir, device=device)
        self.parameters_noramlization_dict = load_normalization_dict(normalization_dict_path)

    def convert_to_onnx(self):
        raise NotImplementedError("ONNX conversion is not supported for this model")

    def generate_output(self, input_embedding, input_latent, expected_parameters, path=DEFAULT_PATH, phrase=DEFAULT_PHRASE, print_output_parameters=False):

        if isinstance(input_embedding, np.ndarray):
            input_embedding = flat_to_torch(input_embedding, EMBEDDING_SHAPE)
        if isinstance(input_latent, np.ndarray):
            input_latent = flat_to_torch(input_latent, LATENT_SHAPE)

        if isinstance(expected_parameters, dict):
            expected_parameters = prepare_parameters(expected_parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)

        modified_embedding, modified_latent = self.modifier_model(expected_parameters, input_embedding, input_latent)

        self.xtts_handler.inference(modified_embedding, modified_latent, phrase=phrase, path=path)
        print(f"Output saved to {path}")

        if print_output_parameters:
            if self.parameters_extractor is None:
                self.parameters_extractor = ParametersExtractor()
            output_parameters = self.parameters_extractor.extract_parameters(path)
            print(f"Output parameters: {output_parameters}")

        return modified_embedding, modified_latent
