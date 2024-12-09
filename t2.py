# Skrypt testowy Kuby do testowania różnych innych skryptów

from embedding_modifier.model_utils.data_generator import DataGenerator
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS

data_generator = DataGenerator()

tensors = data_generator.dimension_latents_dataset_tensors()
print(tensors[0].shape)
print(len(tensors[0][0]))
