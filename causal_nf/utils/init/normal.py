import torch
import torch.nn as nn


def init_normal(std=0.01):
    def _init_normal(module):
        if type(module) == nn.Linear:

            nn.init.normal_(module.weight, mean=0, std=std)
            if isinstance(module.bias, torch.Tensor):
                nn.init.zeros_(module.bias)

    return _init_normal
