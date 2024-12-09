import torch.nn as nn

from utils.parameters_extractor import F0_FLUCTUATIONS_KEY, F0_KEY, F1_KEY, F2_KEY, F3_KEY, GENDER_KEY, HNR_KEY, JITTER_KEY, MEAN_VOICED_SEGMENTS_LENGTH_KEY, SHIMMER_KEY, SKEWNESS_KEY, VOICED_SEGMENTS_PER_SECOND_KEY, ALL_KEYS # REM

# CHOSEN_PARAMETERS_KEYS = [F0_KEY, GENDER_KEY, SKEWNESS_KEY, JITTER_KEY, \
#                            SHIMMER_KEY, HNR_KEY, VOICED_SEGMENTS_PER_SECOND_KEY, \
#                             MEAN_VOICED_SEGMENTS_LENGTH_KEY, F0_FLUCTUATIONS_KEY, \
#                                 F1_KEY, F2_KEY, F3_KEY]

CHOSEN_PARAMETERS_KEYS = ALL_KEYS


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self):
        raise NotImplementedError("Subclasses should implement this method")
