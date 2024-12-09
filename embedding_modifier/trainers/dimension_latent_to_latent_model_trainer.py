import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from embedding_modifier.models.dimension_latent_to_latent_model import DimensionLatentToLatentModel
from embedding_modifier.trainers.model_trainer import ModelTrainer
from settings import DEVICE, DIMENSION_LATENT_TO_LATENT_MODEL_CHECKPOINT_DIR

LEARNING_RATE = 0.005
TENSOR_BOARD_LOGS_DIR = "runs/dimension_latent_to_latent_model"


class DimensionLatentToLatentModelTrainer(ModelTrainer):
    def __init__(self, tensor_board=False, model_version="1"):
        super().__init__(DimensionLatentToLatentModel, TENSOR_BOARD_LOGS_DIR, tensor_board, model_version)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        sh_latents_tensor_train, sh_latents_tensor_validation, sh_latents_tensor_test = \
            self.data_gen.dimension_latents_dataset_tensors()
        
        latents_tensor_train, latents_tensor_validation, latents_tensor_test = \
            self.data_gen.long_latent_datasets_tensors()

        train_dataset = TensorDataset(sh_latents_tensor_train, latents_tensor_train)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(sh_latents_tensor_validation, latents_tensor_validation)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        test_dataset = TensorDataset(sh_latents_tensor_test, latents_tensor_test)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.criterion = torch.nn.MSELoss()  # Funkcja kosztu

    def save_model(self, epoch, path=DIMENSION_LATENT_TO_LATENT_MODEL_CHECKPOINT_DIR):
        return super().save_model(epoch, path)

    # Wczytywanie modelu
    def load_model(self, path=DIMENSION_LATENT_TO_LATENT_MODEL_CHECKPOINT_DIR):
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
