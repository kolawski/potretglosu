import csv
import os

from embedding_modifier.handlers.model_handler import DEFAULT_PHRASE
from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler
from embedding_modifier.handlers.short_latent_to_short_latent_model_handler import ShortLatentToShortLatentModelHandler
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import SHORT_LATENT_SHAPE
from utils.embedding_converter import flat_to_torch
from utils.exceptions import VoiceModifierError
from utils.metrics_retriever import retrieve_metrics
from utils.parameters_extractor import ParametersExtractor
# TODO zrobić utils do wyciągania ssim i mse z 2 wektorów

DEFAULT_PATH = "/app/results/voice_modifier_results/"


class VoiceModifier():
    def __init__(self, handler, model_version, csv_save="voice_modifier_report.csv"):
        self.handler = handler(model_version=model_version)
        self._parameters_extractor = None
        if csv_save is not None:
            self.csv_file_path = DEFAULT_PATH + csv_save
            if not os.path.exists(self.csv_file_path):
                create_headers = True
                mode = 'w'
            else:
                create_headers = False
                mode = 'a'
            self.csv_file = open(self.csv_file_path, mode=mode, newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if create_headers:
                self.csv_writer.writerow(['path', 'parameters_mse', 'short_latent_mse'])

    def __del__(self):
        self.csv_file.close()

    @property
    def parameters_extractor(self):
        if self._parameters_extractor is None:
            self._parameters_extractor = ParametersExtractor()
        return self._parameters_extractor

    def generate_output(self, save_path, expected_parameters, short_latent=None, phrase=DEFAULT_PHRASE):
        recreated_short_latent = None
        if isinstance(self.handler, ParametersToShortLatentModelHandler):
            recreated_short_latent = self.handler.generate_output(expected_parameters, path=save_path, phrase=phrase)
        if isinstance(self.handler, ShortLatentToShortLatentModelHandler):
            if short_latent is None:
                raise VoiceModifierError("Short latent is None")
            recreated_short_latent = self.handler.generate_output(short_latent, expected_parameters, path=save_path, phrase=phrase)

        if recreated_short_latent is None:
            raise VoiceModifierError("Handler is not correct")
        
        parameters = self.parameters_extractor.extract_parameters(save_path)

        parameters = prepare_parameters(parameters, CHOSEN_PARAMETERS_KEYS, self.handler.parameters_noramlization_dict)
        expected_parameters = prepare_parameters(expected_parameters, CHOSEN_PARAMETERS_KEYS, self.handler.parameters_noramlization_dict)

        parameters_mse = retrieve_metrics(expected_parameters, parameters, get_ssim=False)
        if isinstance(self.handler, ShortLatentToShortLatentModelHandler):
            short_latent = flat_to_torch(short_latent, SHORT_LATENT_SHAPE)
            short_latent_mse = retrieve_metrics(short_latent, recreated_short_latent, get_ssim=False)
        else:
            short_latent_mse = "-", "-"

        if self.csv_file is not None:
            self.csv_writer.writerow([save_path, parameters_mse, short_latent_mse])

        return parameters
