import torch.nn as nn

from embedding_modifier.models.model import Model, CHOSEN_PARAMETERS_KEYS

from settings import EMBEDDING_SHAPE, LATENT_SHAPE


class LongLatentModifierModel(Model):
    def __init__(self, number_of_parameters=len(CHOSEN_PARAMETERS_KEYS)):
        super(LongLatentModifierModel, self).__init__()

        self.name = "LongLatentModifierModel"

        # Warstwy dla embedding z uwzględnieniem parametrów (Conv1d zamiast Conv2d)
        self.embedding_transform = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1), # TODO: przemyśleć plusy 2D - czemu czat to sugeruje
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1)
        )
        # Warstwy dla latent z uwzględnieniem parametrów
        self.latent_transform = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )
        # Warstwy włączające parametry
        self.param_embedding_fc = nn.Linear(number_of_parameters, EMBEDDING_SHAPE[1])
        self.param_latent_fc = nn.Linear(number_of_parameters, LATENT_SHAPE[1])

    def forward(self, parameters, embedding, latent):
        # Wpływ parametrów na embedding i latent
        param_embedding = self.param_embedding_fc(parameters).unsqueeze(2)  # [1, 512, 1]
        param_latent = self.param_latent_fc(parameters).unsqueeze(2)        # [1, 32, 1]
        
        # Modyfikacja embedding i latent
        modified_embedding = self.embedding_transform(embedding + param_embedding)
        modified_latent = self.latent_transform(latent + param_latent)
        return modified_embedding, modified_latent




