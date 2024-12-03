# Skrypt testowy Kuby do testowania różnych innych skryptów

from embedding_modifier.model_utils.data_generator import DataGenerator
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS

data_generator = DataGenerator()

data_generator.inference_parameters_datasets_tensors(CHOSEN_PARAMETERS_KEYS)
