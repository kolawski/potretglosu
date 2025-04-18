from embedding_modifier.handlers.model_handler import ModelHandler, DEFAULT_PATH, DEFAULT_PHRASE
from embedding_modifier.models.parameters_to_short_latent_model import ParametersToShortLatentModel
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.handlers.dimension_latent_to_latent_model_handler import DimensionLatentToLatentModelHandler
from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import DEVICE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, SHORT_LATENT_SHAPE


class ParametersToShortLatentModelHandler(ModelHandler):
    def __init__(self, model_version="1", model_dir=PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, device=DEVICE,
                 normalization_dict_path=NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, dimensions_latent_to_latent_model_version="1"):
        super().__init__(ParametersToShortLatentModel, model_version, model_dir, device=device)
        self.parameters_noramlization_dict = load_normalization_dict(normalization_dict_path)
        self.dimensions_latent_to_latent_model_version = dimensions_latent_to_latent_model_version
        self._dimension_latent_to_latent_model_handler = None

    @property
    def dimension_latent_to_latent_model_handler(self):
        if self._dimension_latent_to_latent_model_handler is None:
            self._dimension_latent_to_latent_model_handler = \
                DimensionLatentToLatentModelHandler(self.dimensions_latent_to_latent_model_version)
        return self._dimension_latent_to_latent_model_handler

    def generate_output(self,
                        expected_parameters,
                        path=DEFAULT_PATH,
                        phrase=DEFAULT_PHRASE,
                        return_output_parameters=False,
                        return_short_latent=True,
                        # XTTS inference parameters
                        temperature=0.7,
                        length_penalty=1.0,
                        repetition_penalty=10.0,
                        top_k=50,
                        top_p=0.85,
                        do_sample=True,
                        num_beams=1,
                        speed=1.0,
                        enable_text_splitting=False):
        recreated_short_latent = self.inference(expected_parameters)
        
        results = \
            self.dimension_latent_to_latent_model_handler.\
                generate_output(recreated_short_latent,
                                path,
                                phrase,
                                return_output_parameters,
                                temperature,
                                length_penalty,
                                repetition_penalty,
                                top_k,
                                top_p,
                                do_sample,
                                num_beams,
                                speed,
                                enable_text_splitting
                                )
        
        if return_short_latent:
            return recreated_short_latent
        
        return results  # latent, embedding, output_parameters if return_output_parameters is True
        
    def inference(self, expected_parameters):
        if isinstance(expected_parameters, dict):
            expected_parameters = prepare_parameters(expected_parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)

        recreated_short_latent = self.modifier_model(expected_parameters).reshape(SHORT_LATENT_SHAPE)

        return recreated_short_latent
