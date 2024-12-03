import numpy as np

from utils.embedding_converter import flat_to_torch


def get_only_chosen_parameters(parameters, chosen_parameters):  # TODO maybe move to other module
    return {key: parameters[key] for key in chosen_parameters}


def parameters_to_ndarray(parameters):
    return np.array(list(parameters.values()))


def normalize_parameters(parameters, normalization_dict):
    return {key: parameters[key] / normalization_dict[key] for key in parameters}


def prepare_parameters(parameters, chosen_parameters, normalize=None):
    parameters = get_only_chosen_parameters(parameters, chosen_parameters)
    if normalize is not None:
        parameters = normalize_parameters(parameters, normalize)

    parameters = parameters_to_ndarray(parameters)
    # z konwersją na float tak jak wagi w modelu, unsqueeze żeby było [1, 12] zamiast [12]
    parameters = flat_to_torch(parameters, len(chosen_parameters)).float().unsqueeze(0)
    return parameters
