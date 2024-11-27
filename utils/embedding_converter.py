import numpy as np
import torch

from settings import DEVICE
from utils.exceptions import VoiceModifierError


def flat_to_torch(array, original_shape, device=DEVICE):
    """Converts a flat numpy array to a torch tensor with the original shape and moves it to the specified device

    :param array: flat numpy array to be converted
    :type array: np.ndarray
    :param original_shape: the original shape of the tensor
    :type original_shape: tuple
    :param device: the device to move the tensor to (default is 'cuda')
    :type device: str
    :return: torch tensor with the original shape
    :rtype: torch.Tensor
    """
    tensor = torch.from_numpy(array).reshape(original_shape)
    return tensor.to(device)

def is_flat_array(obj):
    """Checks if the given object is a flat numpy array

    :param obj: object to be checked
    :type obj: any
    :return: True if the object is a flat numpy array, False otherwise
    :rtype: bool
    """
    return isinstance(obj, np.ndarray) and obj.ndim == 1

def is_torch_tensor(obj):
    """Checks if the given object is a torch tensor

    :param obj: object to be checked
    :type obj: any
    :return: True if the object is a torch tensor, False otherwise
    :rtype: bool
    """
    return isinstance(obj, torch.Tensor)

def retrieve_to_flat(*tensors):
    """Converts a list of tensors to their flattened numpy array representations.
    :param tensors: Variable length argument list of tensors. Each tensor can be 
    :type tensors: torch.Tensor or numpy.ndarray
    :returns: A list containing the flattened numpy array representations of the input tensors.
    :rtype: list of numpy.ndarray
    :raises VoiceModifierError: If any of the input tensors are neither a PyTorch tensor nor a flat numpy array.
    """
    tensors_np = []
    for tensor in tensors:
        tensor_np = torch_to_flat(tensor) if is_torch_tensor(tensor) else \
            tensor if is_flat_array(tensor) else None

        if tensor is None:
            raise VoiceModifierError("Invalid data type for retrieval")
        
        tensors_np.append(tensor_np)
    
    return tensors_np

def retrieve_to_tensor(*arrays, original_shape, device='cuda'):
    """Converts a list of arrays to their torch tensor representations with the original shape.

    :param original_shape: the original shape of the tensor
    :type original_shape: tuple
    :param device: _description_, defaults to 'cuda'
    :type device: str, optional
    :raises VoiceModifierError: _description_
    :return: _description_
    :rtype: _type_
    """
    tensors = []
    for array in arrays:
        tensor = flat_to_torch(array, original_shape, device) if is_flat_array(array) else \
            array.reshape(original_shape) if is_torch_tensor(array) else None

        if tensor is None:
            raise VoiceModifierError("Invalid data type for retrieval")
        
        tensors.append(tensor)
    
    return tensors

def torch_to_flat(tensor):
    """Converts a torch tensor to a flat numpy array

    :param tensor: torch tensor to be converted
    :type tensor: torch.Tensor
    :return: flat numpy array
    :rtype: np.ndarray
    """
    return tensor.cpu().detach().numpy().flatten()
