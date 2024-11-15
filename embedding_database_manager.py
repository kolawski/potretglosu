"""Module for managing a database of speaker embeddings, GPT conditionals, audio vectors and sample rates"""
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

from settings import EMBEDDINGS_DB
from utils.audio_sample_converter import audio_to_vec_librosa
from utils.embedding_converter import retrieve_to_flat

AUDIO_KEY = "audio"
SAMPLE_PATH_KEY = "path"
SR_KEY = "sr"
EMBEDDING_KEY = "embedding"
LATENT_KEY = "latent"

class EmbeddingDatabaseManager:
    def __init__(self, db_path=EMBEDDINGS_DB):
        """Constructor

        :param db_path: path to a database file, defaults to EMBEDDINGS_DB
        :type db_path: str, optional
        """
        self._db_path = db_path
        self._dtypes = {
            SAMPLE_PATH_KEY: 'str',
            EMBEDDING_KEY: 'object',
            LATENT_KEY: 'object',
            AUDIO_KEY: 'object',
            SR_KEY: 'int64',
        }
        self._schema = pa.schema([
            (SAMPLE_PATH_KEY, pa.string()),
            (EMBEDDING_KEY, pa.list_(pa.float32())),
            (LATENT_KEY, pa.list_(pa.float32())),
            (AUDIO_KEY, pa.list_(pa.float32())),
            (SR_KEY, pa.int64())
        ])
        if not Path(self._db_path).exists():
            self._dd = self.initialize_dataframe(self._dtypes) # TODO rozgryzc te dtypy jeszcze
        else:
            print(f"Reading database from {self._db_path}")
            self._dd = self.read_from_parquet(self._db_path)

    @property
    def dd(self):
        """Returns copy of the contained Dask DataFrame

        :return: Dask DataFrame
        :rtype: dask.dataframe.core.DataFrame
        """
        return self._dd.copy()

    def add_data(self, embedding, latent, sample_path, parameters):
        """Adds new data to the database

        :param embedding: speaker embedding (XTTS)
        :type embedding: torch.Tensor ([1, 512, 1])
        :param latent: GPT conditional latent (XTTS)
        :type latent: torch.Tensor ([1, 32, 1024])
        :param audio: audio vector retrieved from a file with librosa's load function
        :type audio: numpy.ndarray
        :param sr: sample rate of the audio
        :type sr: int
        """
        
        embedding_np, latent_np = retrieve_to_flat(embedding, latent)
        audio_vector, sr = audio_to_vec_librosa(sample_path)

        new_data = pd.DataFrame({
            SAMPLE_PATH_KEY: [sample_path],
            EMBEDDING_KEY: [embedding_np],
            LATENT_KEY: [latent_np],
            AUDIO_KEY: [audio_vector],
            SR_KEY: [sr]
        } | parameters)

        self._dd = dd.concat([self._dd, dd.from_pandas(new_data, npartitions=1)], axis=0) # TODO co robi axis=0 i npartitions=1 w sumie też

    def delete_data(self, embedding=None, latent=None, sample_path=None):
        """Deletes data from the database

        :param embedding: speaker embedding (XTTS), defaults to None
        :type embedding: torch.Tensor ([1, 512, 1]), optional # TODO or np.array
        :param latent: GPT conditional latent (XTTS), defaults to None
        :type latent: torch.Tensor ([1, 32, 1024]), optional
        :param sample_path: path to an audio file, defaults to None
        :type sample_path: str, optional
        """
        df = self._dd
        if embedding is not None:
            embedding_np = retrieve_to_flat(embedding)
            df = df[df[EMBEDDING_KEY] != embedding_np]
        if latent is not None:
            latent_np = retrieve_to_flat(latent)
            df = df[df[LATENT_KEY] != latent_np]
        if sample_path is not None:
            df = df[df[SAMPLE_PATH_KEY] != sample_path]
        self._dd = df

    def filter_data(self, embedding=None, latent=None, sample_path=None):
        """Returns filtered data from the database. Use only one kwarg at a function call.

        :param embedding: speaker embedding (XTTS), defaults to None
        :type embedding: torch.Tensor ([1, 512, 1]), optional
        :param latent: GPT conditional latent (XTTS), defaults to None
        :type latent: torch.Tensor ([1, 32, 1024]), optional # TODO albo już zapisany będzie numpy
        :param sample_path: path to an audio file, defaults to None
        :type sample_path: str, optional
        :return: filtered data
        :rtype: pandas.DataFrame
        """
        df = self._dd
        if embedding is not None:
            embedding_np = retrieve_to_flat(embedding)
            df = df[df[EMBEDDING_KEY] == embedding_np]
        if latent is not None:
            latent_np = retrieve_to_flat(latent)
            df = df[df[LATENT_KEY] == latent_np]
        if sample_path is not None:
            df = df[df[SAMPLE_PATH_KEY] == sample_path]
        return df.compute()
    
    def get_all_values_from_column(self, key=EMBEDDING_KEY):
        """Returns all unique values for a key (ex. embeddings) from the database

        :return: unique values for a key, defaults to EMBEDDING_KEY
        :rtype: pandas.Series, optional
        """
        return self._dd[key].compute()
        
    @staticmethod
    def initialize_dataframe(dtypes):
        """Initializes a Dask DataFrame with specified data types

        :param dtypes: data types for columns
        :type dtypes: dict
        :return: new Dask DataFrame with specified data types
        :rtype: dask.dataframe.core.DataFrame
        """
        return dd.from_pandas(pd.DataFrame(columns=dtypes.keys()).astype(dtypes), npartitions=1) # TODO czy tu potrzebne te kwargi?

    @staticmethod
    def read_from_parquet(file_path):
        """Reads a Dask DataFrame from a Parquet file with specified data types

        :param file_path: path to a Parquet file
        :type file_path: str
        :return: Dask DataFrame created from a Parquet file
        :rtype: dask.dataframe.core.DataFrame
        """
        return dd.read_parquet(file_path, engine='pyarrow')

    def save_to_parquet(self):
        """Saves the database to a Parquet file"""
        self._dd.to_parquet(self._db_path, engine='pyarrow', schema=self._schema, write_index=True)
