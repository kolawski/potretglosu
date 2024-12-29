import torch.nn as nn
import torch.nn.functional as F

from embedding_modifier.models.model import Model, CHOSEN_PARAMETERS_KEYS


class ParametersToShortLatentModel(Model):
    def __init__(self, input_size=len(CHOSEN_PARAMETERS_KEYS), output_size=1536, hidden_sizes=[256], dropout=0.05):
        super(ParametersToShortLatentModel, self).__init__()

        self.input_size = input_size
        self.name = "ParametersToShortLatentModel"

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

# Model was tested with different configurations:

### VERSION 1 ###
# hidden_sizes=[64, 256, 512, 1024], dropout=0.2
# LEARNING_RATE = 0.0015

### VERSION 2 ###
# hidden_sizes=[64, 128, 256], dropout=0.1
# LEARNING_RATE = 0.0015

### VERSION 3 ###
# hidden_sizes=[128, 256], dropout=0.05
# LEARNING_RATE = 0.0015

### VERSION 4 ###
# hidden_sizes=[256], relu (not leaky_relu), no dropout
# LEARNING_RATE = 0.0015

### VERSION 5 ###
# hidden_sizes=[256], relu (not leaky_relu), no dropout
# LEARNING_RATE = 0.0005

### VERSION 6 ###
# hidden_sizes=[256], sigmoid, no dropout
# LEARNING_RATE = 0.0005

### VERSION 7 ###
# hidden_sizes=[256], tahn, no dropout
# LEARNING_RATE = 0.0005

### VERSION 8 ###
# hidden_sizes=[512], relu, no dropout
# LEARNING_RATE = 0.0005

### VERSION 9 ###
# hidden_sizes=[128], relu, no dropout
# LEARNING_RATE = 0.0005

### VERSION 10 ###
# hidden_sizes=[256, 512], relu (not leaky_relu), no dropout
# LEARNING_RATE = 0.0005

### VERSION 11 ###
# conv
# hidden_channels=[16, 32, 64], dropout=0.05, relu
# LEARNING_RATE = 0.0005

### VERSION 12 ###
# conv
# hidden_channels=[16], dropout=0.05, relu
# LEARNING_RATE = 0.0005

### VERSION 13 ###
# linear
# hidden_sizes=[], relu, no dropout
# LEARNING_RATE = 0.0005

### VERSION 14 ###
# return to version 5

### VERSION 15 ###
# version 5 + sgd instead of adam

### VERSION 16 ###
# version 5 + RMSprop

### VERSION 17 ###
# version 5 + Adagrad

### VERSION 18 ###
# version 5 + LBFGS

### VERSION 19 ###
# version 5, but on all possible parameters

# If you want to test different configurations, you can change the version number
# in the model_version parameter in the training script, or while initializing
# corresponding trainer or handler model. Please note, that you should also
# change the model configuration to parameters described above, otherwise
# checkpoints may not be compatible.

# można to jeszcze dotrenować w akcji sprawdzając czy wygenerowana próbka ma odpowiednie parametry
# no i rozdwoić i skleić <- eee chyba nie
