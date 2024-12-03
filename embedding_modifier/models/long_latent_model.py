import torch.nn as nn

from embedding_modifier.models.model import Model, CHOSEN_PARAMETERS_KEYS


class LongLatentModifierModel(Model):
    def __init__(self, number_of_parameters=len(CHOSEN_PARAMETERS_KEYS)):
        super(LongLatentModifierModel, self).__init__()
        
        # Warstwy dla embedding
        self.embedding_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Warstwy dla latent (obsługa wymiaru [1, 32, 1024])
        self.latent_transform = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),  # Przetwarza wymiar cech (32)
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )
        
        # Warstwy włączające parametry
        self.param_embedding_fc = nn.Linear(number_of_parameters, 512)
        self.param_latent_fc = nn.Linear(number_of_parameters, 32 * 1024)

    def forward(self, parameters, embedding, latent):
        # Wpływ parametrów na embedding i latent
        param_embedding = self.param_embedding_fc(parameters).unsqueeze(2)  # [1, 512, 1]
        param_latent = self.param_latent_fc(parameters).view(1, 32, 1024)   # Dopasowanie do kształtu latentu
        
        # Modyfikacja embedding i latent
        embedding_input = (embedding + param_embedding).squeeze(2)          # [1, 512]
        latent_input = latent + param_latent                                # [1, 32, 1024]
        
        # Przetwarzanie embedding
        modified_embedding = self.embedding_transform(embedding_input).unsqueeze(2)  # [1, 512, 1]
        
        # Przetwarzanie latent (1024 jako wymiar czasowy)
        modified_latent = self.latent_transform(latent_input)  # [1, 32, 1024]
        
        return modified_embedding, modified_latent



