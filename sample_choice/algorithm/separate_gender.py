from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
from utils.parameters_extractor import GENDER_KEY
#from utils.xtts_handler import XTTSHandler
from settings import SAMPLE_PATH_DB_KEY

edb = EmbeddingDatabaseManager()
psldb = ParametersShortLatentsDatabaseManager()
# pomiędzy bazami danych trzeba przechodzić po ścieżkach - niestety trzeba przechowywać te ścieżki razem z embeddingami/latentami przy liczeniu
# albo w słownikach, albo w osobnych obiektach utworzonej do tego celu dataclass
latents = edb.get_all_values_from_column(LATENT_KEY)
paths = edb.get_all_values_from_column(SAMPLE_PATH_DB_KEY)


# random_record = edb.get_random_record()
# path = random_record[SAMPLE_PATH_DB_KEY]
# print(path)

man_indexes = []
women_indexes = []

for i in range(len(latents)):
    latent_i = latents.iloc[i]
    path_i = paths.iloc[i]
    print(path_i)
    print(i)
    try:
        gender = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path_i)[GENDER_KEY]
    except IndexError:
        print(f"IndexError: {i}")
        continue
    if float(gender) > 0.5:
        man_indexes.append(i)
    else:
        women_indexes.append(i)

print(f"Man indexes: {man_indexes}")
print(f"Women indexes: {women_indexes}")

# print(f"random path: {path}")
# embedding = random_record[EMBEDDING_KEY]
# latent = random_record[LATENT_KEY]
#
# gender_for_embedding = psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path)[GENDER_KEY]
#
# print(f"embedding: {embedding}, gender_for_embedding: {gender_for_embedding}")
#
# xtts = XTTSHandler()
# xtts.inference(embedding, latent, path=f"/app/shared/inference_parameters_test_default_1.wav")
# xtts.inference(embedding, latent, path=f"/app/shared/inference_parameters_test_speed_1_5.wav", speed=1.5)