from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from settings import SAMPLE_PATH_DB_KEY

paths_to_remove = ['/app/Resources/ready_audio_samples/common_voice_pl_20603313.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603151.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20600462.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603275.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603464.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603511.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603335.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20602810.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603522.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603514.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603389.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20602880.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603120.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603375.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603241.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603528.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20602940.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603509.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20602965.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603499.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20602988.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603444.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603456.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603513.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603515.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603260.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603209.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20602790.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20595585.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20594754.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603502.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603130.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603481.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603461.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603178.wav']

edb = EmbeddingDatabaseManager()
# pdb = ParametersDatabaseManager()

record_count = edb.get_number_of_records()
print(f"Records in edb: {record_count}")

# record_count = pdb.dd.shape[0].compute()
# print(f"Records in pdb: {record_count}")

for path_to_remove in paths_to_remove:
    print(f"Removing {path_to_remove} from databases")
    edb.delete_record_by_key(SAMPLE_PATH_DB_KEY, path_to_remove)
    # pdb.delete_record_by_key(SAMPLE_PATH_DB_KEY, path_to_remove)

edb.repartition(10)
edb.save_to_parquet()
# pdb.save_to_parquet()

del edb
# del pdb

edb = EmbeddingDatabaseManager()
# pdb = ParametersDatabaseManager()

record_count = edb.get_number_of_records()
print(f"Records in edb: {record_count}")

# record_count = pdb.dd.shape[0].compute()
# print(f"Records in pdb: {record_count}")
