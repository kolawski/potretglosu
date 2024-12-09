import torch

from settings import DEVICE
from embedding_modifier.model_utils.common import find_latest_model_path
from utils.xtts_handler import XTTSHandler

DEFAULT_PATH = "/app/results/result.wav"
DEFAULT_PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."


class ModelHandler:
    def __init__(self, model_type, model_version, model_dir,
                 is_model_dir_checkpoint_path=False, device=DEVICE):
        self.modifier_model = model_type()
        self.modifier_model.to(device)
        if is_model_dir_checkpoint_path is False:
            model_dir = f"{model_dir}/{model_version}"
            model_dir = find_latest_model_path(model_dir)
        self.modifier_model.load_state_dict(torch.load(model_dir)['model_state_dict'])
        self._xtts_handler = None
        self.parameters_extractor = None

    @property
    def xtts_handler(self):
        if self._xtts_handler is None:
            self._xtts_handler = XTTSHandler()
        return self._xtts_handler

    def generate_output(self):
        pass

    
