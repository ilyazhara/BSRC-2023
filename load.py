import torch


def load(model_path, scale):
    return torch.load(model_path, map_location=torch.device("cpu")) if model_path else None
