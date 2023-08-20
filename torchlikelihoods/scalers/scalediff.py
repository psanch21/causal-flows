import torch
import torch.nn as nn

from .base import BaseScaler


class ScaleDiffScaler(BaseScaler):
    def __init__(self):
        self.scale_ = None

    def non_linearity(self):
        return nn.Identity()

    def _fit_params(self, x, dims=None):
        assert dims is None
        x_diff = torch.diff(x)
        params = {"scale_": x_diff.std()}
        return params

    def aggregate_param(self, name, param):
        if name == "scale_":
            param_new = torch.mean(param, dim=0)
        else:
            raise NotImplementedError

        return param_new

    def fit_manual(self):
        self.scale_ = 1.0

    def _transform(self, x, inplace):
        return x / self.scale_

    def _inverse_transform(self, x_norm, inplace):
        return x_norm * self.scale_
