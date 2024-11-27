import os

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from embedding_modifier.data_generator import DataGenerator
from embedding_modifier.modifier_model import ModifierModel
from embedding_modifier.parameters_retriever import ParametersRetriever
from settings import DEVICE, MODEL_CHECKPOINT_PATH, EMBEDDING_SHAPE, LATENT_SHAPE
from utils.embedding_converter import flat_to_torch
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
LEARNING_RATE = 0.003

class ModelTrainer:
    def __init__(self):
        self.writer = SummaryWriter("runs/modifier_model")
        self.model = ModifierModel()
        self.model.to(DEVICE)
        print(f"Model device: {self.model.device}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.data_gen = DataGenerator()
        self.parameters_retriever = ParametersRetriever()

    def _get_only_chosen_parameters(self, parameters):
        return {key: parameters[key] for key in CHOSEN_PARAMETERS_KEYS}

    def _run_tensor_board(self):
        os.system("tensorboard --logdir=runs --port=8050 &")

    def loss_function(self, modified_embedding, modified_latent, original_embedding, original_latent, original_parameters):
        predicted_parameters = self.parameters_retriever.retrieve_parameters(modified_embedding, modified_latent)

        # Strata parametrów
        param_loss = nn.MSELoss()(predicted_parameters, original_parameters)

        # Odległość między wektorami
        embedding_distance = nn.MSELoss()(modified_embedding, original_embedding)
        latent_distance = nn.MSELoss()(modified_latent, original_latent)

        # Dynamiczne ważenie
        total_loss = param_loss + embedding_distance * 0.1 + latent_distance * 0.1
        return total_loss

    # Zapisywanie modelu
    def save_model(self, epoch, path=MODEL_CHECKPOINT_PATH):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    # Wczytywanie modelu
    def load_model(self, path=MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    
    def train(self, epochs=100):
        try:
            start_epoch = self.load_model()
        except FileNotFoundError:
            start_epoch = 0

        for epoch in range(start_epoch, epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Pobierz dane
            embedding, latent = self.data_gen.random_embedding_latent()
            parameters = self._get_only_chosen_parameters(self.data_gen.random_parameters())

            # ONE OD RAZU TUTAJ IDĄ DO CUDA - CZY TAK JEST OK?
            embedding = flat_to_torch(embedding, EMBEDDING_SHAPE)
            latent = flat_to_torch(latent, LATENT_SHAPE)
            parameters = flat_to_torch(parameters, len(CHOSEN_PARAMETERS_KEYS))

            # Forward pass
            modified_embedding, modified_latent = self.model(parameters, embedding, latent)

            # PYTANIE CZY Z MODELU WYJDĄ TENSORY CZY WEKTORY - PODOBNO TENSORY i bardzo możliwe że trzeba będzie je przenieśc na gpu

            # Oblicz stratę
            loss = self.loss_function(modified_embedding, modified_latent, embedding, latent, parameters)

            # Backward pass i optymalizacja
            loss.backward()
            self.optimizer.step()

            # Logowanie do TensorBoard
            self.writer.add_scalar("Loss/train", loss.item(), epoch)

            # Wyświetlanie logów
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                self.save_model(epoch)
