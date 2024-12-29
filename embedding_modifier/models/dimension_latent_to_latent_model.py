import torch.nn as nn
import torch.nn.functional as F

from embedding_modifier.models.model import Model


class DimensionLatentToLatentModel(Model):
    def __init__(self, input_size=1024, output_size=32*1024, hidden_sizes=[8192], dropout=0.05):
        super(DimensionLatentToLatentModel, self).__init__()

        self.input_size = input_size
        self.name = "DimensionLatentToLatentModel"

        # Hidden layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        # self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # Hidden layers for + activation function
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            # x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return x
    

### VERSION 1 ###
# hidden_sizes=[8192]

### VERSION 2 ###
# hidden_sizes=[4096, 8192, 16384]
# model weigth ~ 8GB

### VERSION 3 ###
# hidden_sizes=[2048, 8192]

### VERSION 4 ###
# hidden_sizes=[]


# można byłoby też porównać jak bardzo zdeformowany jest latent po kompresji i przejściu przez model B
# i też względem tego jak jest zdeformowany po powieleniu wymiarów - może nie ma żadnej różnicy
