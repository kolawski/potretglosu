import numpy as np
import torch

from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager, SHORT_LATENT_KEY
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from settings import DEVICE, EMBEDDING_SHAPE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, SAMPLE_PATH_DB_KEY
from utils.exceptions import VoiceModifierError

TENSORS_SAVE_DIR = "/app/Resources/configs"


class DataGenerator:
    def __init__(self):
        self._edb = None
        self._pdb = None
        self._psldb = None

    @property
    def edb(self):
        if self._edb is None:
            self._edb = EmbeddingDatabaseManager()
        return self._edb

    @property
    def pdb(self):
        if self._pdb is None:
            self._pdb = ParametersDatabaseManager()
        return self._pdb

    @property
    def psldb(self):
        if self._psldb is None:
            self._psldb = ParametersShortLatentsDatabaseManager()
        return self._psldb

    def parameters_normalization_dict_pdb(self, keys):
        return self.pdb.get_maximal_values(keys)
    
    def parameters_normalization_dict_psldb(self, keys):
        return self.psldb.get_maximal_values(keys)

    def random_embedding_latent(self):
        record = self.edb.get_random_record()
        return record[EMBEDDING_KEY], record[LATENT_KEY]
    
    def random_fake_parameters(self):
        return self.pdb.get_fake_record()
    
    def random_inference_parameters(self):
        return self.psldb.get_random_record()
    
    def coherent_parameters_embedding_latent(self):
        edb_record = self.edb.get_random_record()
        embedding = edb_record[EMBEDDING_KEY]
        latent = edb_record[LATENT_KEY]
        path = edb_record[SAMPLE_PATH_DB_KEY]
        pdb_record = self.pdb.get_record_by_key(SAMPLE_PATH_DB_KEY, path)
        parameters = pdb_record.to_dict()
        return parameters, embedding, latent
    
    def dimension_latents_dataset_tensors(self,
                                          train_ratio=0.8,
                                          validation_ratio=0.1,
                                          test_ratio=0.1,
                                          short_latent_key=SHORT_LATENT_KEY,
                                          device=DEVICE,
                                          embedding_size=EMBEDDING_SHAPE[1]):
        sh_latents_tensor_train, sh_latents_tensor_validation, sh_latents_tensor_test = \
            self.short_latent_datasets_tensors(train_ratio, validation_ratio, test_ratio, short_latent_key, device)
        
        sh_latents_tensor_train = torch.stack([tensor[:-embedding_size] for tensor in sh_latents_tensor_train])
        sh_latents_tensor_validation = torch.stack([tensor[:-embedding_size] for tensor in sh_latents_tensor_validation])
        sh_latents_tensor_test = torch.stack([tensor[:-embedding_size] for tensor in sh_latents_tensor_test])

        return sh_latents_tensor_train, sh_latents_tensor_validation, sh_latents_tensor_test
    
    def short_latent_and_random_params_datasets_tensors(self,
                                                        chosen_parameters_keys,
                                                        train_ratio=0.8,
                                                        validation_ratio=0.1,
                                                        test_ratio=0.1,
                                                        noise_chance=0.3,
                                                        noise_min=0.85,
                                                        noise_max=1.15,
                                                        latents_repeat=2,
                                                        short_latent_key=SHORT_LATENT_KEY,
                                                        normalization_dict_path=NORMALIZATION_INFERENCE_PARAMS_DICT_PATH,
                                                        device=DEVICE,
                                                        save=False,
                                                        save_dir=TENSORS_SAVE_DIR,
                                                        load=True):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise VoiceModifierError("Sum of ratios should be equal to 1")
        
        if load and save:
            raise VoiceModifierError("Load and save cannot be both True")
        
        if load:
            print("Loading tensors from saved files - all parameters are ignored")
            sh_latents_with_params_tensor_train = torch.load(f"{save_dir}/sh_latents_with_params_tensor_train.pt").to(device)
            sh_latents_with_params_tensor_validation = torch.load(f"{save_dir}/sh_latents_with_params_tensor_validation.pt").to(device)
            sh_latents_with_params_tensor_test = torch.load(f"{save_dir}/sh_latents_with_params_tensor_test.pt").to(device)
            params_tensor_train = torch.load(f"{save_dir}/params_tensor_train.pt").to(device)
            params_tensor_validation = torch.load(f"{save_dir}/params_tensor_validation.pt").to(device)
            params_tensor_test = torch.load(f"{save_dir}/params_tensor_test.pt").to(device)

            return sh_latents_with_params_tensor_train, sh_latents_with_params_tensor_validation, sh_latents_with_params_tensor_test, \
                params_tensor_train, params_tensor_validation, params_tensor_test

        sh_latents = np.array(self.psldb.get_all_values_from_column(short_latent_key).values)
        sh_latents = np.repeat(sh_latents, latents_repeat)

        sh_latents_with_params = []
        params = []
        for sh_latent in sh_latents:
            random_params = self.random_inference_parameters()
            random_params = prepare_parameters(random_params, chosen_parameters_keys,
                                               load_normalization_dict(normalization_dict_path), to_tensor=False)

            if np.random.rand() < noise_chance:
                noise = np.random.uniform(noise_min, noise_max, size=random_params.shape)
                random_params = random_params * noise
                random_params = np.clip(random_params, -1, 1)

            params.append(random_params)

            combined = np.hstack((sh_latent, random_params))
            sh_latents_with_params.append(combined)
        
        sh_latents_with_params = np.array(sh_latents_with_params)
        params = np.array(params)

        number_of_records = len(sh_latents_with_params)
        
        train_end = int(train_ratio * number_of_records)
        validation_end = train_end + int(validation_ratio * number_of_records)

        sh_latents_with_params_train = sh_latents_with_params[:train_end]
        sh_latents_with_params_validation = sh_latents_with_params[train_end:validation_end]
        sh_latents_with_params_test = sh_latents_with_params[validation_end:]

        sh_latents_with_params_tensor_train = \
            torch.tensor(sh_latents_with_params_train, dtype=torch.float32).to(device)
        sh_latents_with_params_tensor_validation = \
            torch.tensor(sh_latents_with_params_validation, dtype=torch.float32).to(device)
        sh_latents_with_params_tensor_test = \
            torch.tensor(sh_latents_with_params_test, dtype=torch.float32).to(device)
        
        params_tensor_train = torch.tensor(params[:train_end], dtype=torch.float32).to(device)
        params_tensor_validation = torch.tensor(params[train_end:validation_end], dtype=torch.float32).to(device)
        params_tensor_test = torch.tensor(params[validation_end:], dtype=torch.float32).to(device)

        if save:
            torch.save(sh_latents_with_params_tensor_train, f"{save_dir}/sh_latents_with_params_tensor_train.pt")
            torch.save(sh_latents_with_params_tensor_validation, f"{save_dir}/sh_latents_with_params_tensor_validation.pt")
            torch.save(sh_latents_with_params_tensor_test, f"{save_dir}/sh_latents_with_params_tensor_test.pt")
            torch.save(params_tensor_train, f"{save_dir}/params_tensor_train.pt")
            torch.save(params_tensor_validation, f"{save_dir}/params_tensor_validation.pt")
            torch.save(params_tensor_test, f"{save_dir}/params_tensor_test.pt")

        return sh_latents_with_params_tensor_train, sh_latents_with_params_tensor_validation, sh_latents_with_params_tensor_test, \
            params_tensor_train, params_tensor_validation, params_tensor_test
        
    def short_latent_datasets_tensors(self,
                                      train_ratio=0.8,
                                      validation_ratio=0.1,
                                      test_ratio=0.1,
                                      short_latent_key=SHORT_LATENT_KEY,
                                      device=DEVICE):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise VoiceModifierError("Sum of ratios should be equal to 1")

        sh_latents = np.array(self.psldb.get_all_values_from_column(short_latent_key).values)
        number_of_records = len(sh_latents)
        
        train_end = int(train_ratio * number_of_records)
        validation_end = train_end + int(validation_ratio * number_of_records)

        sh_latents_train = sh_latents[:train_end]
        sh_latents_validation = sh_latents[train_end:validation_end]
        sh_latents_test = sh_latents[validation_end:]

        sh_latents_tensor_train = torch.tensor(sh_latents_train, dtype=torch.float32).to(device)
        sh_latents_tensor_validation = torch.tensor(sh_latents_validation, dtype=torch.float32).to(device)
        sh_latents_tensor_test = torch.tensor(sh_latents_test, dtype=torch.float32).to(device)

        return sh_latents_tensor_train, sh_latents_tensor_validation, sh_latents_tensor_test
    
    def inference_parameters_datasets_tensors(self,
                                              chosen_parameters_keys,
                                              train_ratio=0.8,
                                              validation_ratio=0.1,
                                              test_ratio=0.1,
                                              normalization_dict_path=NORMALIZATION_INFERENCE_PARAMS_DICT_PATH):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise VoiceModifierError("Sum of ratios should be equal to 1")

        parameters_noramlization_dict = load_normalization_dict(normalization_dict_path)
        parameters = []

        for _, record in self.psldb.dd.iterrows():
            parameters.append(prepare_parameters(record, chosen_parameters_keys, parameters_noramlization_dict))

        number_of_records = len(parameters)
        train_end = int(train_ratio * number_of_records)
        validation_end = train_end + int(validation_ratio * number_of_records)

        parameters_train = parameters[:train_end]
        parameters_validation = parameters[train_end:validation_end]
        parameters_test = parameters[validation_end:]

        parameters_tensor_train = torch.cat(parameters_train, dim=0)
        parameters_tensor_validation = torch.cat(parameters_validation, dim=0)
        parameters_tensor_test = torch.cat(parameters_test, dim=0)

        return parameters_tensor_train, parameters_tensor_validation, parameters_tensor_test
    
    def long_latent_datasets_tensors(self,
                                      train_ratio=0.8,
                                      validation_ratio=0.1,
                                      test_ratio=0.1,
                                      latent_key=LATENT_KEY,
                                      device=DEVICE):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise VoiceModifierError("Sum of ratios should be equal to 1")

        latents = np.array(self.edb.get_all_values_from_column(latent_key).values)
        number_of_records = len(latents)
        
        train_end = int(train_ratio * number_of_records)
        validation_end = train_end + int(validation_ratio * number_of_records)

        latents_train = latents[:train_end]
        latents_validation = latents[train_end:validation_end]
        latents_test = latents[validation_end:]

        latents_tensor_train = torch.tensor(latents_train, dtype=torch.float32).to(device)
        latents_tensor_validation = torch.tensor(latents_validation, dtype=torch.float32).to(device)
        latents_tensor_test = torch.tensor(latents_test, dtype=torch.float32).to(device)

        return latents_tensor_train, latents_tensor_validation, latents_tensor_test

# bierzemy parametry, do nich doklejamy sh_lat no i w zasadzie nie output datasetu nie ma, bo będzie inferencja
# w takiej sytuacji dataloader bez datasetu, tylko z tensora
# lub dataset z inputem i outputem będącym parametrami wejściowymi, ale sam decyduje jak przetwarzany jest loss, a nie
# porównują się automatycznie, z resztą output modelu jest 1536, a mój oczekiwany - parametry - byłyby wtedy 12 tylko
