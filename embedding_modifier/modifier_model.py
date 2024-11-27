import torch.nn as nn

class ModifierModel(nn.Module):
    def __init__(self):
        super(ModifierModel, self).__init__()
        # Warstwy dla embedding z uwzględnieniem parametrów
        self.embedding_transform = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        )
        # Warstwy dla latent z uwzględnieniem parametrów
        self.latent_transform = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        )
        # Warstwy włączające parametry
        self.param_embedding_fc = nn.Linear(40, 512)
        self.param_latent_fc = nn.Linear(40, 32)

    def forward(self, parameters, embedding, latent):
        # Wpływ parametrów na embedding i latent
        param_embedding = self.param_embedding_fc(parameters).unsqueeze(2).unsqueeze(3)
        param_latent = self.param_latent_fc(parameters).unsqueeze(2).unsqueeze(3)
        
        # Modyfikacja embedding i latent
        modified_embedding = self.embedding_transform(embedding + param_embedding)
        modified_latent = self.latent_transform(latent + param_latent)
        return modified_embedding, modified_latent
