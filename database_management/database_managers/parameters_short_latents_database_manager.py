"""Generates database with calculated parameters of an audio file"""
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

from database_management.utils.database_manager_utils import initialize_dataframe, read_from_parquet
from database_management.database_managers.database_manager import DatabaseManager
from utils.parameters_extractor import ALL_KEYS
from settings import PARAMETERS_SHORT_LATENTS_DB, SAMPLE_PATH_DB_KEY

SHORT_LATENT_KEY = 'short_latent'


class ParametersShortLatentsDatabaseManager(DatabaseManager):
    def __init__(self, db_path=PARAMETERS_SHORT_LATENTS_DB, keys=ALL_KEYS):
        """Constructor

        :param db_path: path to a database file, defaults to PARAMETERS_DB
        :type db_path: str, optional
        :param keys: list of parameter keys to be included in the database (float parameters), defaults to ALL_KEYS
        :type keys: list
        """
        self._db_path = db_path
        self._dtypes = {SAMPLE_PATH_DB_KEY: 'str', SHORT_LATENT_KEY: 'object', **{key: 'float64' for key in keys}}

        self._schema = pa.schema([(SAMPLE_PATH_DB_KEY, pa.string()), (SHORT_LATENT_KEY, pa.list_(pa.float32()))] + \
                                 [(key, pa.float64()) for key in keys])
        if not Path(self._db_path).exists():
            self._dd = initialize_dataframe(self._dtypes)
        else:
            print(f"Reading database from {self._db_path}")
            self._dd = read_from_parquet(self._db_path)
    
    def add_data(self, sample_path, short_latent, parameters):
        """Adds new data to the database

        :param sample_path: path to the audio sample
        :type sample_path: str
        :param parameters: dictionary containing calculated parameters of the audio file
        :type parameters: dict
        """
        
        new_data = pd.DataFrame({
            SAMPLE_PATH_DB_KEY: [sample_path],
            SHORT_LATENT_KEY: [short_latent],
            **parameters
        })

        self._dd = dd.concat([self._dd, dd.from_pandas(new_data, npartitions=1)], axis=0)
