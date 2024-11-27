"""Abstract parent class dor some database managers"""
from abc import ABC, abstractmethod

import numpy as np


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
        return self._dd # TODO przemyślec czy copy czy nie copy

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
    
    def get_fake_record(self):  #TODO przetestować
        """Select a random row from each column in the partition and return it as a dictionary.

        This method iterates over each column in the Dask DataFrame `_dd`, computes the column to bring it into memory,
        and then selects a random element from each column. The selected elements are combined into a dictionary.

        :return: A dictionary containing randomly selected elements from each column in the partition.
        :rtype: dict
        """
        return {col: np.random.choice(self._dd[col].compute()) for col in self._dd.columns}
    
    def get_random_record(self):
        """Returns a random record from the database

        :return: random record
        :rtype: pandas.Series
        """
        #TODO: do przetestowania - wydaje się ok
        return self._dd.sample(frac=1).compute().sample(n=1).iloc[0]
    
    def get_record_by_key(self, key, value):
        """Returns a record from the database with a given key-value pair

        :param key: key to search for
        :type key: str
        :param value: value to search for
        :type value: any
        :return: record with the given key-value pair
        :rtype: pandas.Series
        """
        return self._dd[self._dd[key] == value].compute().iloc[0]
    
    def repartition(self, partitions):
        """Repartitions the database

        :param partitions: number of partitions
        :type partitions: int
        """
        self._dd = self._dd.repartition(npartitions=partitions)

    def save_to_parquet(self, path=None):
        """Saves the database to a Parquet file

        :param path: path to save the Parquet file, defaults to the database path
        :type path: str, optional
        """
        if path is None:
            path = self._db_path

        self._dd.to_parquet(path, engine='pyarrow', schema=self._schema) # TODO write_index=True?
        print(f"Saved database to {path}")
