"""Generates database with calculated TSNE"""
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import numpy as np
from sklearn.manifold import TSNE

from embedding_database_manager import EmbeddingDatabaseManager
from settings import TSNE_DB


SAMPLE_PATH_KEY = "path"
EMBEDDING_TSNE_KEY = "embedding_tsne"
LATENT_TSNE_KEY = "latent_tsne"

class TsneDatabaseManager:
    def __init__(self, read_from_file=False):
        if read_from_file:
            self._dd = EmbeddingDatabaseManager.read_from_parquet(TSNE_DB) # TODO może u siebie implementacja byłaby ładniejsza
        else:
            self._dd = self.generate()

        print(self._dd.head())

    @property
    def dd(self):
        return self._dd.copy()
    
    @staticmethod
    def generate(): #TODO: docstringi
        db_manager = EmbeddingDatabaseManager()
        _dd = db_manager.dd

        embedding_np = np.vstack(_dd['embedding'].compute().to_list())
        latent_np = np.vstack(_dd['latent'].compute().to_list())

        tsne = TSNE(n_components=3, random_state=42,
                    perplexity=min(30, len(embedding_np) - 1, len(latent_np) - 1))
        embedding_tsne = tsne.fit_transform(embedding_np)
        latent_tsne = tsne.fit_transform(latent_np)

        tsne_df = pd.DataFrame({
            SAMPLE_PATH_KEY: _dd[SAMPLE_PATH_KEY].compute(),
            EMBEDDING_TSNE_KEY: embedding_tsne.tolist(),
            LATENT_TSNE_KEY: latent_tsne.tolist()
        })

        new_dd = dd.from_pandas(tsne_df, npartitions=_dd.npartitions) # TODO: czy tu nie podać dtypes

        new_schema = pa.schema([
            (SAMPLE_PATH_KEY, pa.string()),
            (EMBEDDING_TSNE_KEY, pa.list_(pa.float32())),
            (LATENT_TSNE_KEY, pa.list_(pa.float32()))
        ])

        new_dd.to_parquet(TSNE_DB, engine='pyarrow', schema=new_schema, write_index=True)
        return new_dd
    
    def get_path_from_embedding_tsne(self, embedding_tsne):
        return self._dd[self._dd[EMBEDDING_TSNE_KEY] == embedding_tsne][SAMPLE_PATH_KEY].compute()
    
    def get_path_from_latent_tsne(self, latent_tsne):
        return self._dd[self._dd[LATENT_TSNE_KEY] == latent_tsne][SAMPLE_PATH_KEY].compute()
