from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from settings import SAMPLE_PATH_DB_KEY

class DataGenerator:
    def __init__(self):
        self.edb = EmbeddingDatabaseManager()
        self.pdb = ParametersDatabaseManager()

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




        