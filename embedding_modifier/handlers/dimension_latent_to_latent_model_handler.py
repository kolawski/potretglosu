
import numpy as np

from embedding_modifier.handlers.model_handler import ModelHandler, DEFAULT_PATH, DEFAULT_PHRASE
from embedding_modifier.models.dimension_latent_to_latent_model import DimensionLatentToLatentModel
from settings import DIMENSION_LATENT_TO_LATENT_MODEL_CHECKPOINT_DIR, DEVICE, EMBEDDING_SHAPE, LATENT_SHAPE, SHORT_LATENT_SHAPE
from utils.embedding_converter import flat_to_torch, torch_to_flat
from utils.parameters_extractor import ParametersExtractor


class DimensionLatentToLatentModelHandler(ModelHandler):
    def __init__(self, model_version="1", model_dir=DIMENSION_LATENT_TO_LATENT_MODEL_CHECKPOINT_DIR, device=DEVICE):
        super().__init__(DimensionLatentToLatentModel, model_version, model_dir, device=device)

    def generate_output(self,
                        short_latent,
                        path=DEFAULT_PATH,
                        phrase=DEFAULT_PHRASE,
                        return_output_parameters=False,
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

        recreated_latent, embedding = self.inference(short_latent, enforce_tensor_output=True)

        self.xtts_handler.inference(embedding=embedding, latent=recreated_latent, path=path, phrase=phrase,
                                    temperature=temperature, length_penalty=length_penalty,
                                    repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p,
                                    do_sample=do_sample, num_beams=num_beams, speed=speed,
                                    enable_text_splitting=enable_text_splitting)

        if return_output_parameters:
            if self.parameters_extractor is None:
                self.parameters_extractor = ParametersExtractor()
            output_parameters = self.parameters_extractor.extract_parameters(path)
            return recreated_latent, embedding, output_parameters

        return recreated_latent, embedding

    def inference(self, short_latent, enforce_tensor_output=False):
        working_on_np_arrays = isinstance(short_latent, np.ndarray)
        if working_on_np_arrays:
            short_latent = flat_to_torch(short_latent, SHORT_LATENT_SHAPE)

        embedding = short_latent[-512:]
        short_latent = short_latent[:1024]

        recreated_latent = self.modifier_model(short_latent)

        if working_on_np_arrays and enforce_tensor_output is False:
            return torch_to_flat(recreated_latent), torch_to_flat(embedding)
        
        recreated_latent = recreated_latent.view(LATENT_SHAPE)
        embedding = embedding.view(EMBEDDING_SHAPE)

        return recreated_latent, embedding
