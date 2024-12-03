import torch.nn as nn
import torch.nn.functional as F

from embedding_modifier.models.model import Model, CHOSEN_PARAMETERS_KEYS


class ParametersToShortLatentModel(Model):
    def __init__(self, input_size=len(CHOSEN_PARAMETERS_KEYS), output_size=1536, hidden_sizes=[64, 256, 512, 1024], dropout=0.2):
        super(ParametersToShortLatentModel, self).__init__()
        
        # Warstwy ukryte
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Warstwa wyjściowa
        self.output_layer = nn.Linear(prev_size, output_size)

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # Przejście przez warstwy ukryte z aktywacją ReLU
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x)
            x = self.dropout(x)
        
        # Warstwa wyjściowa
        x = self.output_layer(x)
        return x

# # Tworzenie modelu
# input_size = 12         # Liczba parametrów wejściowych
# output_size = 1536      # Długość wektora wyjściowego
# hidden_sizes = [64, 256, 512, 1024]  # Rozmiary warstw ukrytych (można dostosować)

# model = ParamToVectorModel(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)

# # Wydruk architektury modelu
# print(model)

# przejrzeć ten model i go dostosować
# dorobić trening modelu
# zrobić klasę rodzica dla trainerów
# dodać odpowiednie funkcję do data_generator lub/i parameters retriever
# dodać wektor normalizacji do nowych parametrów
# wytrenować model

# zrobić nowy folder w embedding_modifier dla modelu short latent to long latent
# zrobić jakiś handler, któy by to obsługiwał

# zrobić trzeci model liczący przesunięcia (architektura II z ważeniem losowym)


# jak będzie czas, to może nowe histogramy dla nowych parametrów wygenerować, byłyby bardziej stabilne pewnie
