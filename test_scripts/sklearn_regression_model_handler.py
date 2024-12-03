import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from database_management.database_managers.embedding_database_manager import EmbeddingDatabaseManager, EMBEDDING_KEY, LATENT_KEY
from database_management.database_managers.parameters_database_manager import ParametersDatabaseManager
from embedding_modifier.model_utils.parameters_utils import get_only_chosen_parameters, normalize_parameters, parameters_to_ndarray
from settings import SAMPLE_PATH_DB_KEY
from utils.exceptions import VoiceModifierError
from utils.parameters_extractor import (
    F0_KEY,
    GENDER_KEY,
    SKEWNESS_KEY,
    JITTER_KEY,
    SHIMMER_KEY,
    HNR_KEY,
    VOICED_SEGMENTS_PER_SECOND_KEY,
    MEAN_VOICED_SEGMENTS_LENGTH_KEY,
    F0_FLUCTUATIONS_KEY,
    F1_KEY,
    F2_KEY,
    F3_KEY
)

CHOSEN_PARAMETERS_KEYS = [F0_KEY, GENDER_KEY, SKEWNESS_KEY, JITTER_KEY, \
                           SHIMMER_KEY, HNR_KEY, VOICED_SEGMENTS_PER_SECOND_KEY, \
                            MEAN_VOICED_SEGMENTS_LENGTH_KEY, F0_FLUCTUATIONS_KEY, \
                                F1_KEY, F2_KEY, F3_KEY]

MODEL_SAVE_PATH = "/app/Resources/model_files/scikit_regression.pkl"

class SklearnRegressionModelHandler:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None
        self.edb = EmbeddingDatabaseManager()
        self.pdb = ParametersDatabaseManager()
        self.embeddings_and_latents = []
        self.parameters = []
        self._parameters_normalization_dict = self.pdb.get_maximal_values(CHOSEN_PARAMETERS_KEYS)

        self.X_test = None
        self.Y_test = None

    def get_data(self):
        embeddings_dd = self.edb.dd
        for _, series in embeddings_dd.iterrows():
            path = series[SAMPLE_PATH_DB_KEY]
            embedding = series[EMBEDDING_KEY]
            latent = series[LATENT_KEY]
            combined_vector = np.concatenate((embedding, latent))
            self.embeddings_and_latents.append(combined_vector)
            all_params = \
                self.pdb.get_record_by_key(SAMPLE_PATH_DB_KEY, path).to_dict()
            
            parameters = get_only_chosen_parameters(all_params, CHOSEN_PARAMETERS_KEYS)
            parameters = normalize_parameters(parameters, self._parameters_normalization_dict)
            parameters = parameters_to_ndarray(parameters)
            for param in parameters:
                if np.isnan(param):
                    print(f"NaN parameter in {path}")
                    break
            self.parameters.append(parameters)

        self.embeddings_and_latents = np.array(self.embeddings_and_latents)
        self.parameters = np.array(self.parameters)

        print(f"Embeddings and latents: {self.embeddings_and_latents}")
        print(f"Parameters: {self.parameters}")

    def train(self):
        X_train, self.X_test, Y_train, self.Y_test = train_test_split(self.parameters[:400], self.embeddings_and_latents[:400],
                                                                      test_size=0.2, random_state=42)

        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = MultiOutputRegressor(base_model)

        self.model.fit(X_train, Y_train)

        print("Model trained successfully.")

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def evaluate(self):
        if self.model is None:
            raise VoiceModifierError("Model is not trained. Please train the model before evaluation.")
        if self.X_test is None or self.Y_test is None:
            _, self.X_test, _, self.Y_test = train_test_split(self.parameters, self.embeddings_and_latents,
                                                              test_size=0.2, random_state=42)
        
        predictions = self.model.predict(self.X_test)

        mse = mean_squared_error(self.Y_test, predictions)
        print(f"Mean Squared Error: {mse}")

        r2 = r2_score(self.Y_test, predictions)
        print(f"R² Score: {r2}")
