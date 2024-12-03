import numpy as np
import torch

from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager, SHORT_LATENT_KEY
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from settings import DEVICE, SAMPLE_PATH_DB_KEY, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH
from utils.exceptions import VoiceModifierError


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
    
    def random_parameters(self):
        return self.pdb.get_fake_record()
    
    def coherent_parameters_embedding_latent(self):
        edb_record = self.edb.get_random_record()
        embedding = edb_record[EMBEDDING_KEY]
        latent = edb_record[LATENT_KEY]
        path = edb_record[SAMPLE_PATH_DB_KEY]
        pdb_record = self.pdb.get_record_by_key(SAMPLE_PATH_DB_KEY, path)
        parameters = pdb_record.to_dict()
        return parameters, embedding, latent
    
    def short_latent_datasets_tensors(self,
                                      train_ratio=0.8,
                                      validation_ratio=0.1,
                                      test_ratio=0.1,
                                      short_latent_key=SHORT_LATENT_KEY,
                                      device=DEVICE):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise VoiceModifierError("Sum of ratios should be equal to 1")

        sh_latents = self.psldb.get_all_values_from_column(short_latent_key).values.tolist()
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

        parameters_train = np.array(parameters[:train_end])
        parameters_validation = np.array(parameters[train_end:validation_end])
        parameters_test = np.array(parameters[validation_end:])

        parameters_tensor_train = torch.cat(parameters_train, dim=0)
        parameters_tensor_validation = torch.cat(parameters_validation, dim=0)
        parameters_tensor_test = torch.cat(parameters_test, dim=0)

        return parameters_tensor_train, parameters_tensor_validation, parameters_tensor_test
