"""Abstract parent class dor some database managers"""
from abc import ABC, abstractmethod


class DatabaseManager(ABC):
    @abstractmethod
    def __init__(self, db_path):
        """Constructor

        :param db_path: path to a database file, defaults to EMBEDDINGS_DB
        :type db_path: str, optional
        """

        self._db_path = db_path
        self._dtypes = None
        self._schema = None
        self._dd = None

    @property
    def dd(self):
        """Returns copy of the contained Dask DataFrame

        :return: Dask DataFrame
        :rtype: dask.dataframe.core.DataFrame
        """
        return self._dd.copy()

    @abstractmethod
    def add_data(self):
        """Adds new data to the database"""
        pass
    
    def get_all_values_from_column(self, key):
        """Returns all unique values for a key (ex. embeddings) from the database

        :return: unique values for a key
        :rtype: pandas.Series
        """
        return self._dd[key].compute()

    def save_to_parquet(self):
        """Saves the database to a Parquet file"""
        self._dd.to_parquet(self._db_path, engine='pyarrow', schema=self._schema, write_index=True)
