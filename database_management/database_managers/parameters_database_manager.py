"""Generates database with calculated parameters of an audio file"""
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

from database_management.utils.database_manager_utils import initialize_dataframe, read_from_parquet
from utils.parameters_extractor import ALL_KEYS
from settings import PARAMETERS_DB


SAMPLE_PATH_KEY = "path"

class ParametersDatabaseManager:
    def __init__(self, db_path=PARAMETERS_DB):
        """Constructor

        :param db_path: path to a database file, defaults to PARAMETERS_DB
        :type db_path: str, optional
        """
        self._db_path = db_path
        self._dtypes = {key: 'float64' for key in ALL_KEYS}

        self._schema = pa.schema([(key, pa.float64()) for key in ALL_KEYS])
        if not Path(self._db_path).exists():
            self._dd = initialize_dataframe(self._dtypes)
        else:
            print(f"Reading database from {self._db_path}")
            self._dd = read_from_parquet(self._db_path)

    @property
    def dd(self):
        """Returns a copy of the Dask DataFrame

        :return: Copy of the Dask DataFrame
        :rtype: dask.dataframe.DataFrame
        """
        return self._dd.copy()
    
    
    def add_data(self, sample_path, parameters):
        """Adds new data to the database

        :param sample_path: path to the audio sample
        :type sample_path: str
        :param parameters: dictionary containing calculated parameters of the audio file
        :type parameters: dict
        """
        
        new_data = pd.DataFrame({
            SAMPLE_PATH_KEY: [sample_path],
            **parameters
        })

        self._dd = dd.concat([self._dd, dd.from_pandas(new_data, npartitions=1)], axis=0)

    def save_to_parquet(self):
        """Saves the database to a Parquet file"""
        self._dd.to_parquet(self._db_path, engine='pyarrow', schema=self._schema, write_index=True)
