import torch

from settings import DEVICE
from embedding_modifier.model_utils.common import find_latest_model_path, load_normalization_dict
from utils.xtts_handler import XTTSHandler

DEFAULT_PATH = "/app/results/result.wav"
DEFAULT_PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."


class ModelHandler:
    def __init__(self, model_type, model_dir, normalization_dict_path,
                 is_model_dir_checkpoint_path=False, device=DEVICE):
        self.modifier_model = model_type()
        self.modifier_model.to(device)
        if not is_model_dir_checkpoint_path:
            model_dir = find_latest_model_path(model_dir)
        self.modifier_model.load_state_dict(torch.load(model_dir)['model_state_dict'])
        self.xtts_handler = XTTSHandler()
        self.parameters_noramlization_dict = load_normalization_dict(normalization_dict_path)

    def generate_output(self):
        pass

    
