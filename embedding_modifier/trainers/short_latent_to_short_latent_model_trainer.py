import csv

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from embedding_modifier.handlers.dimension_latent_to_latent_model_handler import DimensionLatentToLatentModelHandler
from embedding_modifier.models.model import CHOSEN_PARAMETERS_KEYS
from embedding_modifier.models.short_latent_to_short_latent_model import ShortLatentToShortLatentModel
from embedding_modifier.model_utils.common import load_normalization_dict
from embedding_modifier.model_utils.parameters_utils import prepare_parameters
from embedding_modifier.trainers.model_trainer import ModelTrainer
from settings import DEVICE, NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, SHORT_LATENT_SHAPE, SHORT_LATENT_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR
from utils.exceptions import VoiceModifierError
from utils.parameters_extractor import ParametersExtractor
from utils.xtts_handler import XTTSHandler

DISTANCE_LOSS_WEIGHT = 0.8
LEARNING_RATE = 0.005
TENSOR_BOARD_LOGS_DIR = "runs/short_latent_to_short_latent_model"
TMP_OUTPUT_PATH = "/app/tmp/short_latent_to_short_latent_tmp.wav"
CSV_REPORT_DIR = "/app/results/training_data/"
PATAMETERS_LOSS_WEIGTHS = [5, 30, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
ACCEPTED_L2_ERROR = 1
EXCEEDING_L2_ERROR_COST_RATIO = 10
BATCH_SIZE = 1


class ShortLatentToShortLatentModelTrainer(ModelTrainer):
    def __init__(self, tensor_board=False, model_version="1", distance_loss_weight=DISTANCE_LOSS_WEIGHT,
                 normalization_dict_path=NORMALIZATION_INFERENCE_PARAMS_DICT_PATH, parameters_weigths=PATAMETERS_LOSS_WEIGTHS,
                 accepted_l2_error=ACCEPTED_L2_ERROR, exceeding_l2_error_cost_ratio=EXCEEDING_L2_ERROR_COST_RATIO,
                 batch_size=BATCH_SIZE):
        super().__init__(ShortLatentToShortLatentModel, TENSOR_BOARD_LOGS_DIR, tensor_board, model_version)

        self.report_path = F"{CSV_REPORT_DIR}short_latent_to_short_latent_model_{model_version}.csv"
        self.parameters_noramlization_dict = load_normalization_dict(normalization_dict_path)
        self.parameters_weigths = parameters_weigths
        if len(self.parameters_weigths) != len(CHOSEN_PARAMETERS_KEYS):
            raise VoiceModifierError("Parameters weigths must have the same length as CHOSEN_PARAMETERS_KEYS")
        self.parameters_weights = torch.tensor(self.parameters_weigths, dtype=torch.float32, device=DEVICE)
        self.accepted_l2_error = accepted_l2_error
        self.exceeding_l2_error_cost_ratio = exceeding_l2_error_cost_ratio
        self.batch_size = batch_size
        self.distance_loss_weight = distance_loss_weight
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self._dim_to_long_latent_handler = None
        self._parameters_extractor = None
        self._xtts = None

        self.csv_file = open(self.report_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        sh_latents_with_params_tensor_train, sh_latents_with_params_tensor_validation, \
            sh_latents_with_params_tensor_test, params_tensor_train, params_tensor_validation, \
            params_tensor_test = self.data_gen.short_latent_and_random_params_datasets_tensors(CHOSEN_PARAMETERS_KEYS)

        train_dataset = TensorDataset(sh_latents_with_params_tensor_train, params_tensor_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(sh_latents_with_params_tensor_validation, params_tensor_validation)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        test_dataset = TensorDataset(sh_latents_with_params_tensor_test, params_tensor_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.criterion = self.criterion_default  # Funkcja kosztu

    def __del__(self):
        self.csv_file.close()

    @property
    def dim_to_long_latent_handler(self):
        if self._dim_to_long_latent_handler is None:
            self._dim_to_long_latent_handler = DimensionLatentToLatentModelHandler()
        return self._dim_to_long_latent_handler
    
    @property
    def parameters_extractor(self):
        if self._parameters_extractor is None:
            self._parameters_extractor = ParametersExtractor()
        return self._parameters_extractor
    
    @property
    def xtts_handler(self):
        if self._xtts is None:
            self._xtts = XTTSHandler()
        return self._xtts

    def save_model(self, epoch, path=SHORT_LATENT_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR):
        return super().save_model(epoch, path)

    def load_model(self, path=SHORT_LATENT_TO_SHORT_LATENT_MODEL_CHECKPOINT_DIR):
        return super().load_model(path)
    
    def criterion_default(self, output_batch, target_batch, input_batch, sample_count):
        total_loss = 0.0

        for output, target, input_tensor in zip(output_batch, target_batch, input_batch):
            # Generowanie parametrów dla pojedynczego tensora
            self.dim_to_long_latent_handler.generate_output(output, path=TMP_OUTPUT_PATH)
            extracted_parameters = self.parameters_extractor.extract_parameters(TMP_OUTPUT_PATH)
            parameters = prepare_parameters(extracted_parameters, CHOSEN_PARAMETERS_KEYS, requires_grad=True, normalize=self.parameters_noramlization_dict)

            # Obliczanie straty MSE dla parametrów
            mse_loss = torch.nn.MSELoss()(parameters.squeeze() * self.parameters_weights, target.squeeze() * self.parameters_weights)
            print(f"MSE Loss: {mse_loss}")
            
            # Obliczanie straty L2 dla odległości output-input
            l2_loss = torch.nn.functional.mse_loss(output, input_tensor[:SHORT_LATENT_SHAPE[0]])  # Norma L2 = kwadratowa MSE dla pierwszych 1536 współrzędnych
            print(f"L2 Loss: {l2_loss}")

            # self.writer.add_scalar("mse/Checkpoint_Train", mse_loss.item(), sample_count)
            # self.writer.add_scalar("L2_Loss/Checkpoint_Train", l2_loss.item(), sample_count)
            self.csv_writer.writerow([target, parameters])

            # Łączna strata
            # total_loss += mse_loss + self.exceeding_l2_error_cost_ratio * max(0, l2_loss-self.accepted_l2_error)
            total_loss += mse_loss + self.distance_loss_weight * l2_loss
            self.writer.add_scalar("Loss/Checkpoint_Train", l2_loss.item(), sample_count)

        # Średnia strata dla batcha
        avg_loss = total_loss / len(output_batch)
        return avg_loss

    def train_epoch(self, checkpoint_interval=300):
        self.model.train()
        total_train_loss = 0.0
        batch_count = 0
        sample_count = 0  # Liczba przetworzonych próbek w danej epoce

        for inputs, targets in self.train_loader:
            batch_count += 1
            sample_count += len(inputs)

            # Definiujemy closure
            def closure():
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, inputs, sample_count)
                loss.backward()
                return loss

            # Wykonanie kroku optymalizatora z closure
            loss = self.optimizer.step(closure)
            total_train_loss += loss.item()

            # Logowanie i checkpointy co `checkpoint_interval` próbek
            if sample_count % checkpoint_interval < len(inputs):  # Po przekroczeniu progu
                avg_loss_so_far = total_train_loss / batch_count
                print(f"Checkpoint: {sample_count} samples processed, Avg Loss: {avg_loss_so_far:.4f}")
                self.save_model(epoch=f"epoch_partial_{sample_count}")
                self.writer.add_scalar("Loss/Checkpoint_Train", avg_loss_so_far, sample_count)

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

    def train(self, epochs=100, load_model=True, checkpoint_interval=300):
        if load_model:
            try:
                start_epoch = self.load_model()
            except FileNotFoundError:
                start_epoch = 0
        else:
            self.csv_writer.writerow(["expected_params", "output_params"])
            start_epoch = 0

        for epoch in range(start_epoch, epochs):
            print(f"Starting epoch {epoch}...")
            avg_train_loss = self.train_epoch(checkpoint_interval=checkpoint_interval)
            avg_val_loss = self.validate_epoch()

            # Logi do TensorBoard
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

            # Wyświetlanie logów co 10 epok
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
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
