"""A freshly initialized coordinate-conditioned RGB student, with JSON export."""

import torch
from torch import nn


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(6, 64), nn.Linear(64, 64), nn.Linear(64, 3)))

    def forward(self, inputs):
        hidden = torch.relu(self.layers[0](inputs))
        hidden = torch.relu(self.layers[1](hidden))
        return torch.sigmoid(self.layers[2](hidden))

    def export_layers(self):
        result = []
        for index, layer in enumerate(self.layers):
            for parameter in (layer.weight, layer.bias):
                if not torch.isfinite(parameter).all() or (parameter.abs() > 1000).any():
                    raise ValueError("Student parameters violate the finite, absolute-1000 bound")
            result.append({
                "input": layer.in_features,
                "output": layer.out_features,
                "activation": "sigmoid" if index == 2 else "relu",
                "weights": layer.weight.detach().cpu().reshape(-1).tolist(),
                "bias": layer.bias.detach().cpu().tolist(),
            })
        return result
