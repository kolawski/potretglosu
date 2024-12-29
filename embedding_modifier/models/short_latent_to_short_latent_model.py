import torch
import torch.nn as nn

from embedding_modifier.models.model import Model, CHOSEN_PARAMETERS_KEYS
from settings import EMBEDDING_SHAPE, LATENT_SHAPE, SHORT_LATENT_SHAPE


class SubModel(Model):
    def __init__(self, input_size, output_size, hidden_sizes, activation_fn=nn.ReLU, dropout=0.0):
        super(SubModel, self).__init__()
        layers = []
        prev_size = input_size
        
        # Tworzenie warstw ukrytych
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size
        
        # Warstwa wyjściowa
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ShortLatentToShortLatentModel(Model):
    def __init__(self, 
                 param_size=len(CHOSEN_PARAMETERS_KEYS), 
                 latent_size=LATENT_SHAPE[2], 
                 embedding_size=EMBEDDING_SHAPE[1], 
                 latent_hidden_sizes=[512],
                 embedding_hidden_sizes=[256],
                 activation_fn=nn.ReLU, 
                 dropout=0.0):
        super(ShortLatentToShortLatentModel, self).__init__()

        self.input_size = param_size + SHORT_LATENT_SHAPE[0]
        self.name = "ShortLatentToShortLatentModel"
        
        # Podmodel dla latentu
        self.latent_model = SubModel(
            input_size=param_size + latent_size,
            output_size=latent_size,
            hidden_sizes=latent_hidden_sizes,
            activation_fn=activation_fn,
            dropout=dropout
        )
        
        # Podmodel dla embeddingu
        self.embedding_model = SubModel(
            input_size=param_size + embedding_size,
            output_size=embedding_size,
            hidden_sizes=embedding_hidden_sizes,
            activation_fn=activation_fn,
            dropout=dropout
        )

    # training forward
    def forward(self, sh_latent_batch):
        outputs = []
        for sh_latent in sh_latent_batch:  # Iteracja po batchu
            latent = sh_latent[:LATENT_SHAPE[2]]
            embedding = sh_latent[LATENT_SHAPE[2]:EMBEDDING_SHAPE[1] + LATENT_SHAPE[2]]
            params = sh_latent[EMBEDDING_SHAPE[1] + LATENT_SHAPE[2]:]

            latent_input = torch.cat([params, latent], dim=-1)
            embedding_input = torch.cat([params, embedding], dim=-1)

            modified_latent = self.latent_model(latent_input)
            modified_embedding = self.embedding_model(embedding_input)

            output = torch.cat([modified_latent, modified_embedding], dim=-1)
            outputs.append(output)

        return torch.stack(outputs)  # Łączenie wyników w batch
    
    # # usage forward
    # def forward(self, sh_latent):
    #     latent = sh_latent[:LATENT_SHAPE[2]]
    #     embedding = sh_latent[LATENT_SHAPE[2]:EMBEDDING_SHAPE[1] + LATENT_SHAPE[2]]
    #     params = sh_latent[EMBEDDING_SHAPE[1] + LATENT_SHAPE[2]:]

    #     latent_input = torch.cat([params, latent], dim=-1)
    #     embedding_input = torch.cat([params, embedding], dim=-1)

    #     modified_latent = self.latent_model(latent_input)
    #     modified_embedding = self.embedding_model(embedding_input)

    #     return torch.cat([modified_latent, modified_embedding], dim=-1)

    

### VERSION 1 ###
# latent_hidden_sizes=[256, 512]
# embedding_hidden_sizes=[128, 256]

### VERSION 3 ###
# latent_hidden_sizes=[512]
# embedding_hidden_sizes=[256]

### Added normalization ###

### VERSION 4 ###
# latent_hidden_sizes=[2048], 
# embedding_hidden_sizes=[1024],

### VERSION 5 ###
# latent_hidden_sizes=[], 
# embedding_hidden_sizes=[], 

### VERSION 6 ###
# hierarchical weighting - loss penalty if too far, otherwise optimize parameters, set to 1 (chosen from experiments)
# latent_hidden_sizes=[512],
# embedding_hidden_sizes=[256],
# PATAMETERS_LOSS_WEIGTHS = [5, 30, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
# ACCEPTED_L2_ERROR = 1
# EXCEEDING_L2_ERROR_COST_RATIO = 10
# BATCH_SIZE=1
# CHINEESE/DAMAGED output

### VERSION 7 ###
# VERSION 6 and return to weighted loss

# jeszcze zrobić wersję z większą ilością warstw
# ewentualnie wszystkimi parametrami

