from embedding_modifier.data_generator import DataGenerator
from embedding_modifier.parameters_retriever import ParametersRetriever


data_generator = DataGenerator()
parameters_retriever = ParametersRetriever()

for _ in range(10):
    params = data_generator.random_parameters()
    print(params)
    print("---")

embedding, latent = data_generator.random_embedding_latent()
params = parameters_retriever.retrieve_parameters(embedding, latent)

print(f"Params from embedding and latent: {params}")