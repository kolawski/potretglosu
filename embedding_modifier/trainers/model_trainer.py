import os

import torch
from torch.utils.tensorboard import SummaryWriter

from embedding_modifier.model_utils.common import find_latest_model_path
from embedding_modifier.model_utils.data_generator import DataGenerator
from settings import DEVICE


class ModelTrainer:
    def __init__(self, model_type, tensor_board_logs_dir, tensor_board=True):

        if not os.path.exists(tensor_board_logs_dir):
            os.makedirs(tensor_board_logs_dir)

        self.writer = SummaryWriter(tensor_board_logs_dir)
        self.model = model_type()
        self.model.to(DEVICE)
        self.data_gen = DataGenerator()
        if tensor_board:
            self._run_tensor_board()

        self.checkpoint_iteration = 1

    def __del__(self):
        self.writer.close()

    def _run_tensor_board(self):
        os.system("tensorboard --logdir=/app/runs --host=0.0.0.0 --port=8050 &")

    def save_model(self, epoch, dir):
        dir = f"{dir}/checkpoint_{self.checkpoint_iteration}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, dir)
        self.checkpoint_iteration += 1

    def load_model(self, dir):
        path = find_latest_model_path(dir)

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    
    def train(self, epochs=100):
        pass
