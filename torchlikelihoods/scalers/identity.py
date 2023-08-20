import torch.nn as nn

from .base import BaseScaler


class IdentityScaler(BaseScaler):
    def __init__(self):
        return

    def non_linearity(self):
        return nn.Identity()

    def _fit_params(self, x, dims=None):
        return {}

    def aggregate_param(self, name, param):
        return param

    def fit_manual(self):
        return

    def _transform(self, x, inplace):
        return x

    def _inverse_transform(self, x_norm, inplace):
        return x_norm
