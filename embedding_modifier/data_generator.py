from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY

class DataGenerator:
    def __init__(self):
        self.edb = EmbeddingDatabaseManager()
        self.pdb = ParametersDatabaseManager()

    def random_embedding_latent(self):
        record = self.edb.get_random_record()
        return record[EMBEDDING_KEY], record[LATENT_KEY]
    
    def random_parameters(self):
        return self.pdb.get_fake_record()
        