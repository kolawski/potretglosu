import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

from utils.embedding_converter import flat_to_torch


def retrieve_metrics(vector_a, vector_b, get_ssim=True, get_m2e=True):
    if not isinstance(vector_a, torch.Tensor):
        vector_a = flat_to_torch(vector_a)
    if not isinstance(vector_b, torch.Tensor):
        vector_b = flat_to_torch(vector_b)

    output = []
    if get_ssim:
        ssim_val = ssim(vector_a.unsqueeze(1), vector_b.unsqueeze(1))
        print(f"ssim: {ssim_val}")
        output.append(ssim_val.item())
    if get_m2e:
        m2e_val = torch.mean((vector_a - vector_b) ** 2)
        print(f"m2e: {m2e_val}")
        output.append(m2e_val.item())

    return output