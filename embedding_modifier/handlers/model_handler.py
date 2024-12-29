import os

import torch
import torch.onnx 

from settings import DEVICE
from embedding_modifier.model_utils.common import find_latest_model_path
from utils.xtts_handler import XTTSHandler

DEFAULT_PATH = "/app/results/result.wav"
DEFAULT_PHRASE = "Jestem klonem głosu. Mówię ciekawe rzeczy i można mnie dostosować."


class ModelHandler:
    def __init__(self, model_type, model_version, model_dir,
                 is_model_dir_checkpoint_path=False, device=DEVICE):
        self.device = device
        self.modifier_model = model_type()
        self.modifier_model.to(self.device)
        if is_model_dir_checkpoint_path is False:
            model_dir = f"{model_dir}/{model_version}"
            self.model_dir = model_dir
            model_checkpoint_dir = find_latest_model_path(model_dir)
        else:
            self.model_dir = os.path.dirname(model_dir)
        self.modifier_model.load_state_dict(torch.load(model_checkpoint_dir)['model_state_dict'])
        self._xtts_handler = None
        self.parameters_extractor = None

    @property
    def xtts_handler(self):
        if self._xtts_handler is None:
            self._xtts_handler = XTTSHandler()
        return self._xtts_handler
    
    def convert_to_onnx(self):
        # set the model to inference mode 
        self.modifier_model.eval() 

        # Let's create a dummy input tensor  
        dummy_input = torch.randn(1, self.modifier_model.input_size, requires_grad=True).to(self.device) 

        # Export the model   
        torch.onnx.export(self.modifier_model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            f"{self.model_dir}/model.onnx",       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput'], # the model's output names 
            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                    'modelOutput' : {0 : 'batch_size'}}) 
        print(" ") 
        print('Model has been converted to ONNX')

        # set the model back to training mode 
        self.modifier_model.train()

    def generate_output(self):
        pass
