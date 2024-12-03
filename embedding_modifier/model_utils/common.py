import os

import numpy as np


def find_latest_model_path(dir):
        checkpoint_files = [f for f in os.listdir(dir) if f.endswith('.pth')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in the directory {dir}.")
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        return os.path.join(dir, latest_checkpoint)

def load_normalization_dict(path):
    """
    Loads the normalization dictionary from the given path.

    :param path: path to the normalization dictionary
    :type path: str
    :return: normalization dictionary
    :rtype: dict
    """
    parameters_noramlization_dict = np.load(path)
    parameters_noramlization_dict = \
        {key: value.item() for key, value in parameters_noramlization_dict.items()}
    return parameters_noramlization_dict