import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager, SHORT_LATENT_KEY
from embedding_modifier.handlers.dimension_latent_to_latent_model_handler import DimensionLatentToLatentModelHandler
from settings import EMBEDDING_SHAPE, LATENT_SHAPE, SAMPLE_PATH_DB_KEY
from utils.embedding_converter import flat_to_torch
import csv


basic_path = "/app/results/dimension_latent_to_latent_model_tests"

edb = EmbeddingDatabaseManager()
psldb = ParametersShortLatentsDatabaseManager()
model_handler = DimensionLatentToLatentModelHandler(model_version="1")

csv_file_path = f"{basic_path}/report_ssim_m2e.csv"
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["model_recreated_latent_ssim", "repeatedly_recreated_latent_ssim", "model_recreated_latent_m2e", "repeatedly_recreated_latent_m2e"])

    for i in range(250):
        random_record = edb.get_random_record()
        sample_path = random_record[SAMPLE_PATH_DB_KEY]
        original_latent = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)
        short_latent = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, sample_path)[SHORT_LATENT_KEY]

        # path = f"{basic_path}/model_recreated_latent_{i}.wav"
        # recreated_latent, embedding = model_handler.generate_output(short_latent, path=path, print_output_parameters=True)

        recreated_latent, embedding = model_handler.inference(short_latent, enforce_tensor_output=True)

        # path = f"{basic_path}/repeatedly_recreated_latent_{i}.wav"
        repeated_dimension = original_latent.mean(dim=1, keepdim=True).repeat(1, 32, 1)
        # model_handler.xtts_handler.inference(embedding, repeated_dimension, path=path)

        # path = f"{basic_path}/original_model_and_latent_{i}.wav"
        # model_handler.xtts_handler.inference(embedding, original_latent, path=path)

        for latent in (recreated_latent, repeated_dimension):
            if not isinstance(latent, torch.Tensor):
                latent = flat_to_torch(latent, LATENT_SHAPE)

        model_recreated_latent_ssim = ssim(original_latent.unsqueeze(1), recreated_latent.unsqueeze(1))
        print(f"Model recreated latent ssim {i}: {model_recreated_latent_ssim}")
        repeatedly_recreated_latent_ssim = ssim(original_latent.unsqueeze(1), repeated_dimension.unsqueeze(1))
        print(f"Repeatedly recreated latent ssim {i}: {repeatedly_recreated_latent_ssim}")

        model_recreated_latent_m2e = torch.mean((original_latent - recreated_latent) ** 2)
        print(f"Model recreated latent m2e {i}: {model_recreated_latent_m2e}")
        repeatedly_recreated_latent_m2e = torch.mean((original_latent - repeated_dimension) ** 2)
        print(f"Repeatedly recreated latent m2e {i}: {repeatedly_recreated_latent_m2e}")

        writer.writerow([model_recreated_latent_ssim.item(), repeatedly_recreated_latent_ssim.item(), \
                         model_recreated_latent_m2e.item(), repeatedly_recreated_latent_m2e.item()])


# different speakers:
csv_file_path = f"{basic_path}/report_random_speakers_latents_comparison.csv"
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ssim", "mse"])

    for i in range(255):
        random_record_1 = edb.get_random_record()
        original_latent_1 = flat_to_torch(random_record_1[LATENT_KEY], LATENT_SHAPE)

        random_record_2 = edb.get_random_record()
        original_latent_2 = flat_to_torch(random_record_2[LATENT_KEY], LATENT_SHAPE)


        ssim_diff = ssim(original_latent_1.unsqueeze(1), original_latent_2.unsqueeze(1))
        print(f"Smim difference {i}: {ssim_diff}")

        m2e_diff = torch.mean((original_latent_1 - original_latent_2) ** 2)
        print(f"m2e difference {i}: {m2e_diff}")

        writer.writerow([ssim_diff.item(), m2e_diff.item()])

# same speaker:
csv_file_path = f"{basic_path}/report_same_speaker_latents_comparison.csv"
temp_file_path = f"{basic_path}/temp.wav"
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ssim", "mse"])

    for j in range(5):
        random_record = edb.get_random_record()
        original_latent = flat_to_torch(random_record[LATENT_KEY], LATENT_SHAPE)
        embedding = flat_to_torch(random_record[EMBEDDING_KEY], EMBEDDING_SHAPE)

        for i in range(5):
            model_handler.xtts_handler.inference(embedding, original_latent, path=temp_file_path)
            new_latent, new_embedding = model_handler.xtts_handler.compute_latents(temp_file_path)


            ssim_diff = ssim(original_latent.unsqueeze(1), new_latent.unsqueeze(1))
            print(f"Smim difference {i}: {ssim_diff}")

            m2e_diff = torch.mean((original_latent - new_latent) ** 2)
            print(f"m2e difference {i}: {m2e_diff}")

            writer.writerow([ssim_diff.item(), m2e_diff.item()])
