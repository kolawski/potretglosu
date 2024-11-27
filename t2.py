# from test_scripts.parameters_comparators.different_phrases_comparator import DifferentPhrasesComparator
# from test_scripts.parameters_comparators.different_inferences_comparator import DifferentInterfacesComparator
# from test_scripts.parameters_comparators.same_embedding_different_latents_comparator import SameEmbeddingsDifferentLatentsComparator
# from test_scripts.parameters_comparators.same_latent_different_embeddings_comparator import SameLatentDifferentEmbeddingsComparator

# # jeszcze wcześniej znowu utworzyć bazę wszystkich parametrów na nowo
# # i odpalić histogram generator na niej

# # comparator = DifferentPhrasesComparator(iterations=100)
# # comparator.run_comparison()

# # comparator = DifferentInterfacesComparator(iterations=100)
# # comparator.run_comparison()

# # comparator = SameEmbeddingsDifferentLatentsComparator(iterations=100)
# # comparator.run_comparison()

# comparator = SameLatentDifferentEmbeddingsComparator(iterations=100)
# comparator.run_comparison()

##########

from embedding_modifier.data_generator import DataGenerator
from embedding_modifier.parameters_retriever import ParametersRetriever

# dobra trzeba jakoś zrobić żeby się pobierały do modelu tylko te parametry, które on tam ma ustalone
# W TYM CELU TRZEBA PRZETESTOWAĆ jak działają parameters retriever i data generator

data_generator = DataGenerator()
parameters_retriever = ParametersRetriever()

for _ in range(3):
    params = data_generator.random_parameters()
    print(params)
    print("/n")

embedding, latent = data_generator.random_embedding_latent()
params = parameters_retriever.retrieve_parameters(embedding, latent)

print(f"Params from embedding and latent: {params}")



