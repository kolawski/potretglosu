import csv
from tqdm import tqdm

from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler
from embedding_modifier.model_utils.data_generator import DataGenerator
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from utils.metrics_retriever import retrieve_metrics
from utils.parameters_extractor import ParametersExtractor


iterations = 60
wav_temp_path = "/app/results/models_efficiency_tests/A_prim_model_efficiency_temp.wav"
csv_file_path = "/app/results/models_efficiency_tests/model_A_19_parameters_difference_on_parameters_from_test_dataset.csv"

progress_bar = tqdm(total=iterations, desc="Processing")

file = open(csv_file_path, mode='a', newline='')
writer = csv.writer(file)
writer.writerow(["parameters_mse"])

data_gen = DataGenerator()
parameters_tensor_train, parameters_tensor_validation, parameters_tensor_test = \
    data_gen.inference_parameters_datasets_tensors(CHOSEN_PARAMETERS_KEYS)

parameters_tensor_test = parameters_tensor_test[:iterations]

handler_A = ParametersToShortLatentModelHandler(model_version="19")
normalization_dict = handler_A.parameters_noramlization_dict

parameters_extractor = ParametersExtractor()

for i in range(iterations):
    expected_parameters = parameters_tensor_test[i]
    handler_A.generate_output(expected_parameters, path=wav_temp_path)
    retrieved_parameters = parameters_extractor.extract_parameters(wav_temp_path)
    retrieved_parameters = prepare_parameters(retrieved_parameters, CHOSEN_PARAMETERS_KEYS, normalization_dict)
    parameters_mse = retrieve_metrics(expected_parameters, retrieved_parameters, get_ssim=False)
    writer.writerow([parameters_mse])
    progress_bar.update(1)

file.close()

# iterations = 60
# wav_temp_path = "/app/results/models_efficiency_tests/A_prim_model_efficiency_temp.wav"
# csv_file_path = "/app/results/models_efficiency_tests/model_A_19_prim_stability_tests.csv"

# progress_bar = tqdm(total=iterations, desc="Processing")

# file = open(csv_file_path, mode='a', newline='')
# writer = csv.writer(file)
# writer.writerow(['parameters_19_mse','embeddings_model_19_mse', 'latents_model_19_mse', 'latents_model_19_ssim'])

# data_gen = DataGenerator()
# parameters_tensor_train, parameters_tensor_validation, parameters_tensor_test = \
#     data_gen.inference_parameters_datasets_tensors(CHOSEN_PARAMETERS_KEYS)

# parameters_tensor_test = parameters_tensor_test[:iterations]

# handler_A = ParametersToShortLatentModelHandler(model_version="19")
# normalization_dict = handler_A.parameters_noramlization_dict

# parameters_extractor = ParametersExtractor()

# for i in range(iterations):
#     expected_parameters = parameters_tensor_test[i]

#     latent_1, embedding_1 = handler_A.generate_output(expected_parameters, path=wav_temp_path, return_short_latent=False)
#     retrieved_parameters_1 = parameters_extractor.extract_parameters(wav_temp_path)
#     retrieved_parameters_1 = prepare_parameters(retrieved_parameters_1, CHOSEN_PARAMETERS_KEYS, normalization_dict)
    
#     latent_2, embedding_2 = handler_A.generate_output(expected_parameters, path=wav_temp_path, return_short_latent=False)
#     retrieved_parameters_2 = parameters_extractor.extract_parameters(wav_temp_path)
#     retrieved_parameters_2 = prepare_parameters(retrieved_parameters_2, CHOSEN_PARAMETERS_KEYS, normalization_dict)

#     parameters_mse = retrieve_metrics(retrieved_parameters_1, retrieved_parameters_2, get_ssim=False)
#     embeddings_mse = retrieve_metrics(embedding_1, embedding_2, get_ssim=False)
#     latents_ssim, latents_mse = retrieve_metrics(latent_1, latent_2)
#     writer.writerow([parameters_mse, embeddings_mse, latents_mse, latents_ssim])
#     progress_bar.update(1)

# file.close()
