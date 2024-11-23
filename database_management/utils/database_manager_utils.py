"""Utility functions for database management"""
import dask.dataframe as dd
import pandas as pd
import pyarrow


def initialize_dataframe(dtypes):
    """Initializes a Dask DataFrame with specified data types

    :param dtypes: data types for columns
    :type dtypes: dict
    :return: new Dask DataFrame with specified data types
    :rtype: dask.dataframe.core.DataFrame
    """
    return dd.from_pandas(pd.DataFrame(columns=dtypes.keys()).astype(dtypes), npartitions=1) # TODO czy tu potrzebne te kwargi?


def read_from_parquet(file_path):
    """Reads a Dask DataFrame from a Parquet file with specified data types

    :param file_path: path to a Parquet file
    :type file_path: str
    :return: Dask DataFrame created from a Parquet file
    :rtype: dask.dataframe.core.DataFrame
    """
    return dd.read_parquet(file_path, engine='pyarrow')
