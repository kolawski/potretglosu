from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from settings import SAMPLE_PATH_DB_KEY

path_to_remove = "/app/Resources/ready_audio_samples/common_voice_pl_20651405.wav"

edb = EmbeddingDatabaseManager()
pdb = ParametersDatabaseManager()

record_count = edb.dd.shape[0].compute()
print(f"Records in edb: {record_count}")

record_count = pdb.dd.shape[0].compute()
print(f"Records in pdb: {record_count}")

exit()
print(f"Removing {path_to_remove} from databases")
edb.delete_record_by_key(SAMPLE_PATH_DB_KEY, path_to_remove)
pdb.delete_record_by_key(SAMPLE_PATH_DB_KEY, path_to_remove)

edb.save_to_parquet()
pdb.save_to_parquet()

del edb
del pdb

edb = EmbeddingDatabaseManager()
pdb = ParametersDatabaseManager()

record_count = edb.dd.shape[0].compute()
print(f"Records in edb: {record_count}")

record_count = pdb.dd.shape[0].compute()
print(f"Records in pdb: {record_count}")
