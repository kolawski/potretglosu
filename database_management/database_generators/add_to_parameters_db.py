"""Module for retrieving parameters from audio sample and adding it into data base"""
from pathlib import Path

from tqdm import tqdm

from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from settings import SAMPLES_DIRECTORY_PATH
from utils.parameters_extractor import ParametersExtractor


samples_dir = Path(SAMPLES_DIRECTORY_PATH)

file_paths = [str(file) for file in samples_dir.glob('*') if file.is_file()]
progress_bar = tqdm(total=len(file_paths), desc="Processing")

parameters_extractor = ParametersExtractor()
db_manager = ParametersDatabaseManager()

for sample_path in file_paths:
    print(f"Processing {repr(sample_path)}")

    parameters = parameters_extractor.extract_parameters(sample_path)

    db_manager.add_data(sample_path, parameters)
    progress_bar.update(1)

db_manager.repartition(30)
db_manager.save_to_parquet()

print(db_manager._dd.compute().head())
