from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.data_generator import DataGenerator
from embedding_modifier.model_utils.parameters_retriever_from_embedding import ParametersRetrieverFromEmbedding


data_generator = DataGenerator()
# parameters_retriever = ParametersRetrieverFromEmbedding()

# for _ in range(10):
#     params = data_generator.random_fake_parameters()
#     print(params)
#     print("---")

# embedding, latent = data_generator.random_embedding_latent()
# params = parameters_retriever.retrieve_parameters(embedding, latent)

# print(f"Params from embedding and latent: {params}")

sh_latents_with_params_tensor_train, sh_latents_with_params_tensor_validation, \
    sh_latents_with_params_tensor_test, params_tensor_train, params_tensor_validation, \
    params_tensor_test = data_generator.short_latent_and_random_params_datasets_tensors(CHOSEN_PARAMETERS_KEYS, save=True, load=False)

print(sh_latents_with_params_tensor_train.shape)
print(sh_latents_with_params_tensor_validation.shape)
print(sh_latents_with_params_tensor_test.shape)
print(params_tensor_train.shape)
print(params_tensor_validation.shape)
print(params_tensor_test.shape)

print("sh_latents_with_params_tensor_train")
print(sh_latents_with_params_tensor_train[0])
print("params_tensor_train")
print(params_tensor_train[0])
