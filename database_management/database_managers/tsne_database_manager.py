"""Generates database with calculated TSNE"""
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import numpy as np
from sklearn.manifold import TSNE

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager
from database_management.utils.database_manager_utils import read_from_parquet
from settings import TSNE_DB, SAMPLE_PATH_DB_KEY


EMBEDDING_TSNE_KEY = "embedding_tsne"
LATENT_TSNE_KEY = "latent_tsne"

class TsneDatabaseManager:
    def __init__(self, read_from_file=False):
        """
        Initializes the TSNEDatabaseManager.
        :param read_from_file: If True, reads the data from a parquet file. 
                               If False, generates the data. Default is False.
        :type read_from_file: bool
        """
        if read_from_file:
            self._dd = read_from_parquet(TSNE_DB)
        else:
            self._dd = self.generate()

        print(self._dd.head())

    @property
    def dd(self):
        """
        Create a copy of the internal _dd attribute.
        :return: A copy of the _dd attribute.
        :rtype: dict
        """
        return self._dd.copy()
    
    @staticmethod
    def generate():
        """Generates the t-SNE data and saves it to a parquet file."""
        db_manager = EmbeddingDatabaseManager()
        _dd = db_manager.dd

        embedding_np = np.vstack(_dd['embedding'].compute().to_list())
        latent_np = np.vstack(_dd['latent'].compute().to_list())

        tsne = TSNE(n_components=3, random_state=42,
                    perplexity=min(30, len(embedding_np) - 1, len(latent_np) - 1))
        embedding_tsne = tsne.fit_transform(embedding_np)
        latent_tsne = tsne.fit_transform(latent_np)

        tsne_df = pd.DataFrame({
            SAMPLE_PATH_DB_KEY: _dd[SAMPLE_PATH_DB_KEY].compute(),
            EMBEDDING_TSNE_KEY: embedding_tsne.tolist(),
            LATENT_TSNE_KEY: latent_tsne.tolist()
        })

        new_dd = dd.from_pandas(tsne_df, npartitions=_dd.npartitions)

        new_schema = pa.schema([
            (SAMPLE_PATH_DB_KEY, pa.string()),
            (EMBEDDING_TSNE_KEY, pa.list_(pa.float32())),
            (LATENT_TSNE_KEY, pa.list_(pa.float32()))
        ])

        new_dd.to_parquet(TSNE_DB, engine='pyarrow', schema=new_schema, write_index=True)
        return new_dd
    
    def get_path_from_embedding_tsne(self, embedding_tsne):
        """
        Retrieve the sample path corresponding to a given t-SNE embedding.
        :param embedding_tsne: The t-SNE embedding for which to retrieve the sample path.
        :type embedding_tsne: Any
        :return: The sample path corresponding to the given t-SNE embedding.
        :rtype: Any
        """
        return self._dd[self._dd[EMBEDDING_TSNE_KEY] == embedding_tsne][SAMPLE_PATH_DB_KEY].compute()
    
    def get_path_from_latent_tsne(self, latent_tsne):
        """
        Retrieve the sample path corresponding to a given latent t-SNE value.
        :param latent_tsne: The latent t-SNE value to search for.
        :type latent_tsne: float
        :return: The sample path associated with the given latent t-SNE value.
        :rtype: str
        """
        return self._dd[self._dd[LATENT_TSNE_KEY] == latent_tsne][SAMPLE_PATH_DB_KEY].compute()
