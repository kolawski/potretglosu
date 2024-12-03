from embedding_modifier.model_utils.data_generator import DataGenerator
from embedding_modifier.model_utils.parameters_retriever_from_embedding import ParametersRetrieverFromEmbedding


data_generator = DataGenerator()
parameters_retriever = ParametersRetrieverFromEmbedding()

for _ in range(10):
    params = data_generator.random_parameters()
    print(params)
    print("---")

embedding, latent = data_generator.random_embedding_latent()
params = parameters_retriever.retrieve_parameters(embedding, latent)

print(f"Params from embedding and latent: {params}")