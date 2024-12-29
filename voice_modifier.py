from database_management.database_managers.parameters_short_latents_database_manager import ParametersShortLatentsDatabaseManager
from embedding_modifier.handlers.parameters_to_short_latent_model_handler import ParametersToShortLatentModelHandler, DEFAULT_PHRASE
from settings import SAMPLE_PATH_DB_KEY
from utils.parameters_extractor import ParametersExtractor

DEFAULT_TEMP_PATH = "/app/tmp/voice_changer_tmp.wav"
PARAMETERS_TO_SHORT_LATENT_DEFAULT_MODEL_VERSION = "5"  # "5" is best 12-parameters version


class VoiceModifier:
    def __init__(self, parameters_to_short_latent_model_version=PARAMETERS_TO_SHORT_LATENT_DEFAULT_MODEL_VERSION):
        self.parameters_to_short_latent_model_version = parameters_to_short_latent_model_version
        
        self._psldb = None
        self._handler = None
        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = ParametersExtractor()
        return self._extractor

    @property
    def psldb(self):
        if self._psldb is None:
            self._psldb = ParametersShortLatentsDatabaseManager()
        return self._psldb
    
    @property
    def handler(self):
        if self._handler is None:
            self._handler = ParametersToShortLatentModelHandler(model_version=self.parameters_to_short_latent_model_version)
        return self._handler

    def retrieve_parameters_from_embedding_latent(self, embedding, latent):
        self.handler.xtts_handler.inference(embedding, latent, path=DEFAULT_TEMP_PATH)
        return self.extractor.extract_parameters(DEFAULT_TEMP_PATH)

    def retrieve_parameters_from_path(self, path):
        record = self.psldb.get_record_by_key(SAMPLE_PATH_DB_KEY, path)
        return record.to_dict()

    def generate_sample_from_parameters(self,
                                        parameters,
                                        path=DEFAULT_TEMP_PATH,
                                        phrase=DEFAULT_PHRASE,
                                        # XTTS inference parameters
                                        temperature=0.7,
                                        length_penalty=1.0,
                                        repetition_penalty=10.0,
                                        top_k=50,
                                        top_p=0.85,
                                        do_sample=True,
                                        num_beams=1,
                                        speed=1.0,
                                        enable_text_splitting=False):
        return self.handler.generate_output(expected_parameters=parameters,
                                            path=path, phrase=phrase,
                                            return_short_latent=False,
                                            return_output_parameters=True,
                                            temperature=temperature,
                                            length_penalty=length_penalty,
                                            repetition_penalty=repetition_penalty,
                                            top_k=top_k,
                                            top_p=top_p,
                                            do_sample=do_sample,
                                            num_beams=num_beams,
                                            speed=speed,
                                            enable_text_splitting=enable_text_splitting)
