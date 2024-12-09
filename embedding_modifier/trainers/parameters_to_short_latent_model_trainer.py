import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.models.parameters_to_short_latent_model import ParametersToShortLatentModel
from embedding_modifier.trainers.model_trainer import ModelTrainer
from settings import DEVICE, PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH

LEARNING_RATE = 0.005
TENSOR_BOARD_LOGS_DIR = "runs/parameters_to_short_latent_model"


class ParametersToShortLatentModelTrainer(ModelTrainer):
    def __init__(self, tensor_board=False, save_normalization_dict=True, model_version="1"):
        super().__init__(ParametersToShortLatentModel, TENSOR_BOARD_LOGS_DIR, tensor_board, model_version)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.parameters_noramlization_dict = self.data_gen.parameters_normalization_dict_psldb(CHOSEN_PARAMETERS_KEYS)
        if save_normalization_dict:
            np.savez(NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, **self.parameters_noramlization_dict)

        sh_latents_tensor_train, sh_latents_tensor_validation, sh_latents_tensor_test = \
            self.data_gen.short_latent_datasets_tensors()
        
        parameters_tensor_train, parameters_tensor_validation, parameters_tensor_test = \
            self.data_gen.inference_parameters_datasets_tensors(CHOSEN_PARAMETERS_KEYS)

        train_dataset = TensorDataset(parameters_tensor_train, sh_latents_tensor_train)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(parameters_tensor_validation, sh_latents_tensor_validation)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        test_dataset = TensorDataset(parameters_tensor_test, sh_latents_tensor_test)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.criterion = torch.nn.MSELoss()  # Funkcja kosztu

    def save_model(self, epoch, path=PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR):
        return super().save_model(epoch, path)

    # Wczytywanie modelu
    def load_model(self, path=PARAMETERS_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR):
        return super().load_model(path)
    
    def train_epoch(self):
        self.model.train()
        total_train_loss = 0.0

        for inputs, targets in self.train_loader:
            # Definiujemy closure
            def closure():
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                return loss

            # Wykonanie kroku optymalizatora z closure
            loss = self.optimizer.step(closure)
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(self.train_loader)
        return avg_train_loss

    def validate_epoch(self):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss

    def train(self, epochs=100, load_model=True):
        if load_model:
            try:
                start_epoch = self.load_model()
            except FileNotFoundError:
                start_epoch = 0
        else:
            start_epoch = 0

        for epoch in range(start_epoch, epochs):
            avg_train_loss = self.train_epoch()
            avg_val_loss = self.validate_epoch()

            # Logi do TensorBoard
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

            # Wyświetlanie logów
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            if epoch % 30 == 0:
                self.save_model(epoch)

        print("Finished training")


    def test(self):
        self.model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(self.test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        self.writer.add_scalar("Loss/Test", avg_test_loss)
