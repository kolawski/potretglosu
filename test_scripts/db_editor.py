from pathlib import Path

from tqdm import tqdm

from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager, SAMPLE_PATH_KEY
from settings import SAMPLES_DIRECTORY_PATH
from utils.parameters_extractor import ParametersExtractor, F0_KEY


samples_dir = Path(SAMPLES_DIRECTORY_PATH)

file_paths = [str(file) for file in samples_dir.glob('*') if file.is_file()]
progress_bar = tqdm(total=len(file_paths), desc="Processing")

parameters_extractor = ParametersExtractor()
db_manager = ParametersDatabaseManager("/app/Resources/databases/2_f0_test_all.parquet", [F0_KEY])

print(db_manager.get_all_values_from_column(F0_KEY))

# for sample_path in file_paths:
#     print(f"Processing {repr(sample_path)}")

#     parameters = parameters_extractor.extract_parameters(sample_path)

#     db_manager.add_data(sample_path, parameters)
#     progress_bar.update(1)

# db_manager.save_to_parquet()

# print(db_manager._dd.compute().head())

dd = db_manager.dd
filtered_samples = ["/app/Resources/ready_audio_samples/common_voice_pl_40726223.wav", '/app/Resources/ready_audio_samples/common_voice_pl_20603393.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20604058.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605264.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605717.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607408.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20617549.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20622224.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20624532.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20631222.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20637277.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20637395.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20650579.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20653436.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20668769.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20775930.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20776503.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20945998.wav', '/app/Resources/ready_audio_samples/common_voice_pl_21098294.wav', '/app/Resources/ready_audio_samples/common_voice_pl_21775656.wav']
# dd = dd[~dd[SAMPLE_PATH_KEY].isin(filtered_samples)]

filtered_samples = dd[dd[F0_KEY] > 500][SAMPLE_PATH_KEY].compute().to_list()
print(filtered_samples)
# input("Press Enter to continue...")
# big_f0 = ['/app/Resources/ready_audio_samples/common_voice_pl_20603120.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603151.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603393.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603515.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603604.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603773.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603792.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603987.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20604902.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605264.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605498.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605563.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605618.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20606261.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607075.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607774.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607938.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607947.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20608485.wav']
# big_f0_2 = ['/app/Resources/ready_audio_samples/common_voice_pl_20603151.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603393.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603792.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603987.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20604058.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20604902.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605264.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605717.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607408.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607938.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607947.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20608485.wav']
# big_f0_4 = ['/app/Resources/ready_audio_samples/common_voice_pl_20603393.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607947.wav']
# big_f0_5 = ['/app/Resources/ready_audio_samples/common_voice_pl_20603393.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20603792.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20604058.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605264.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20605717.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607408.wav', '/app/Resources/ready_audio_samples/common_voice_pl_20607947.wav']

# print(f"big_f0: {len(big_f0)}")
# print(f"big_f0_2: {len(big_f0_2)}")
# print(f"big_f0_4: {len(big_f0_4)}")
# print(f"big_f0_5: {len(big_f0_5)}")
