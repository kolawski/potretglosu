import numpy as np
import torch
import torch.optim as optim

from utils.embedding_converter import flat_to_torch
from embedding_modifier.models.long_latent_model import LongLatentModifierModel
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.parameters_retriever_from_embedding import ParametersRetrieverFromEmbedding
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from embedding_modifier.trainers.model_trainer import ModelTrainer
from settings import LONG_LATENT_MODEL_CHECKPOINT_DIR, EMBEDDING_SHAPE, LATENT_SHAPE, NORMALIZATION_DICT_PATH

LEARNING_RATE = 0.0015
TENSOR_BOARD_LOGS_DIR = "runs/long_latent_model"


class LongLatentModelTrainer(ModelTrainer):
    def __init__(self, tensor_board=True, save_normalization_dict=True, model_version="1"):
        super().__init__(LongLatentModifierModel, TENSOR_BOARD_LOGS_DIR, tensor_board, model_version)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.parameters_retriever = ParametersRetrieverFromEmbedding()
        self.parameters_noramlization_dict = self.data_gen.parameters_normalization_dict_pdb(CHOSEN_PARAMETERS_KEYS)
        if save_normalization_dict:
            np.savez(NORMALIZATION_DICT_PATH, **self.parameters_noramlization_dict)

    # def loss_function(self, modified_embedding, modified_latent, original_embedding, original_latent, original_parameters):
    #     predicted_parameters = self.parameters_retriever.retrieve_parameters(modified_embedding, modified_latent)
    #     predicted_parameters = prepare_parameters(predicted_parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)
    #     # TODO: pierwsza próbka jest tak zła, że: [None 0.17 9.4464 nan nan nan 0.0 0.0 0.0 0.0 0.0 0.0]
    #     # albo zmapować none/nan na jakieś bardzo złe wartości, albo jakiś pretraining z innym lossem i inputem na początek

    #     # Strata parametrów
    #     # param_loss = nn.MSELoss()(predicted_parameters, original_parameters)

    #     param_loss = torch.norm(predicted_parameters - original_parameters, p=2)

    #     # Odległość między wektorami
    #     # embedding_distance = nn.MSELoss()(modified_embedding, original_embedding)
    #     # latent_distance = nn.MSELoss()(modified_latent, original_latent)

    #     # Dynamiczne ważenie
    #     # total_loss = param_loss + embedding_distance * 0.1 + latent_distance * 0.1
    #     # return total_loss
    #     return param_loss

    # Zapisywanie modelu
    def save_model(self, epoch, dir=LONG_LATENT_MODEL_CHECKPOINT_DIR):
        return super().save_model(epoch, dir)

    # Wczytywanie modelu
    def load_model(self, dir=LONG_LATENT_MODEL_CHECKPOINT_DIR):
        return super().load_model(dir)
    
    # def train(self, epochs=100, load_model=True):
    #     if load_model:
    #         try:
    #             start_epoch = self.load_model()
    #         except FileNotFoundError:
    #             start_epoch = 0
    #     else:
    #         start_epoch = 0

    #     for epoch in range(start_epoch, epochs):
    #         self.model.train()
    #         self.optimizer.zero_grad()

    #         # Pobierz dane
    #         embedding, latent = self.data_gen.random_embedding_latent()
    #         parameters = prepare_parameters(self.data_gen.random_parameters(), CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)

    #         # ONE OD RAZU TUTAJ IDĄ DO CUDA - CZY TAK JEST OK?
    #         embedding = flat_to_torch(embedding, EMBEDDING_SHAPE)
    #         latent = flat_to_torch(latent, LATENT_SHAPE)

    #         # Forward pass
    #         modified_embedding, modified_latent = self.model(parameters, embedding, latent)

    #         # PYTANIE CZY Z MODELU WYJDĄ TENSORY CZY WEKTORY - PODOBNO TENSORY i bardzo możliwe że trzeba będzie je przenieśc na gpu

    #         # Oblicz stratę
    #         loss = self.loss_function(modified_embedding, modified_latent, embedding, latent, parameters)

    #         # Backward pass i optymalizacja
    #         loss.backward()
    #         self.optimizer.step()

    #         # Logowanie do TensorBoard
    #         self.writer.add_scalar("Loss/train", loss.item(), epoch)
    #         self.writer.flush()

    #         # Wyświetlanie logów
    #         if epoch % 10 == 0:
    #             print(f"Epoch {epoch}, Loss: {loss.item()}")
    #             self.save_model(epoch)

    #     print("Finished training")

    def loss_function(self, modified_embedding, modified_latent, original_embedding, original_latent, original_parameters):
        # Odległość między wektorami
        embedding_distance = torch.norm(modified_embedding - self.current_expected_embedding, p=2)
        latent_distance = torch.norm(modified_latent - self.current_expected_latent, p=2)

        # TODO: ta funkcja kosztu w pretrainerze musi być stosunkowo duża, żeby model był w stanie się dotrenować już właściwym treningiem (chyba?)
        total_loss = embedding_distance + latent_distance
        return total_loss
 
    def train(self, epochs=100, load_model=True):
        if load_model:
            try:
                start_epoch = self.load_model()
            except FileNotFoundError:
                start_epoch = 0
        else:
            start_epoch = 0

        for epoch in range(start_epoch, epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Pobierz dane
            embedding, latent = self.data_gen.random_embedding_latent()
            parameters, coherent_embedding, coherent_latent = \
                self.data_gen.coherent_parameters_embedding_latent()
            parameters = prepare_parameters(parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_dict)

            self.current_expected_embedding = flat_to_torch(coherent_embedding, EMBEDDING_SHAPE)
            self.current_expected_latent = flat_to_torch(coherent_latent, LATENT_SHAPE)

            embedding = flat_to_torch(embedding, EMBEDDING_SHAPE)
            latent = flat_to_torch(latent, LATENT_SHAPE)

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
            self.writer.flush()

            # Wyświetlanie logów
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                self.save_model(epoch)

        print("Finished training")
