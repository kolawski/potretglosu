# from data_visualizer import DataVisualizer
# from tsne_database_manager import TsneDatabaseManager
from utils.parameters_extractor import ParametersExtractor

#najpierw odpalić add_to_embedding_db.py (a wcześniej to ją i tsne usunąć)
# potem tsne manager, żeby policzył swoją
#no i visualizer

# tsne_man = TsneDatabaseManager()

# dv = DataVisualizer()

# dv.visualize()

############################################

ext = ParametersExtractor()

dir_path = "/app/Resources/ready_audio_samples/"
file_path1 = f"{dir_path}common_voice_pl_21643510.wav"
file_path2 = f"{dir_path}common_voice_pl_20606171.wav"
file_path3 = f"{dir_path}common_voice_pl_20613853.wav"

files = [file_path1, file_path2, file_path3]

for file in files:
    res = ext.extract_parameters(file)
    print(f"{file}: {res}")

############################################

# import opensmile

# # Utwórz instancję OpenSMILE
# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.emobase
#     # feature_level=opensmile.FeatureLevel.Functionals
# )

# # Przetwórz plik audio i wyświetl dostępne kolumny
# features = smile.process_file(file_path1)
# print(features.columns)



