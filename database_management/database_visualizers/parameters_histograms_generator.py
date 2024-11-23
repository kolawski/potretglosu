"""Generates histograms for all parameters in the database."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from utils.parameters_extractor import ALL_KEYS


HISTOGRAMS_SAVE_DIRECTORY_PATH = "/app/results/histograms/"

Path(HISTOGRAMS_SAVE_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)

pdb = ParametersDatabaseManager()

for key in ALL_KEYS:
    series = pdb.get_all_values_from_column(key)
    mean_value = series.mean()
    std_deviation = series.std()
    title = f"{key}, \mu={mean_value:.2f}, \sigma={std_deviation:.2f}"
    sns.histplot(data=series, bins='auto', kde=True, color="orange")
    plt.title(title)
    plt.savefig(HISTOGRAMS_SAVE_DIRECTORY_PATH + key + ".png")
    plt.close()
    print(f"Saved histogram for {key}")
