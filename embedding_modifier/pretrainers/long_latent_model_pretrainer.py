import torch

from embedding_modifier.trainers.long_latent_model_trainer import LongLatentModelTrainer
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from settings import EMBEDDING_SHAPE, LATENT_SHAPE
from utils.embedding_converter import flat_to_torch

LEARNING_RATE = 0.003


class LongLatentModelPretrainer(LongLatentModelTrainer):
    def __init__(self, tensor_board=True):
        super().__init__(tensor_board)
        self.current_expected_latent = None
        self.current_expected_embedding = None

    def loss_function(self, modified_embedding, modified_latent, original_embedding, original_latent, original_parameters):
        # Odległość między wektorami
        embedding_distance = torch.norm(modified_embedding - self.current_expected_embedding, p=2)
        latent_distance = torch.norm(modified_latent - self.current_expected_latent, p=2)

        # TODO: ta funkcja kosztu w pretrainerze musi być stosunkowo duża, żeby model był w stanie się dotrenować już właściwym treningiem (chyba?)
        total_loss = embedding_distance + latent_distance
        return total_loss
 
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
            parameters, coherent_embedding, coherent_latent = \
                self.data_gen.coherent_parameters_embedding_latent()
            parameters = prepare_parameters(parameters, CHOSEN_PARAMETERS_KEYS, self.parameters_noramlization_vector)

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
