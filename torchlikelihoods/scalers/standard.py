import torch
import torch.nn as nn

from .base import BaseScaler


class StandardScaler(BaseScaler):
    def __init__(self):
        self.mu_ = None
        self.scale_ = None

    def non_linearity(self):
        return nn.Identity()

    def _fit_params(self, x, dims=None):
        if isinstance(dims, tuple):
            mu_ = x.mean(dims[0], keepdims=True)
            scale_ = x.std(dims, keepdims=True)
            for dim in dims[1:]:
                mu_ = mu_.mean(dim, keepdims=True)
        elif isinstance(dims, list):
            raise Exception("dims should be None or a tuple!")
        else:
            mu_ = x.mean()
            scale_ = x.std()
        params = {"mu_": mu_, "scale_": scale_}
        return params

    def aggregate_param(self, name, param):
        if name == "mu_":
            param_new = torch.mean(param, dim=0)
        elif name == "scale_":
            param_new = torch.mean(param, dim=0)
        else:
            raise NotImplementedError

        return param_new

    def fit_manual(self):
        self.mu_ = 0.0
        self.scale_ = 1.0

    def _transform(self, x, inplace):
        return (x - self.mu_) / self.scale_

    def _inverse_transform(self, x_norm, inplace):
        return x_norm * self.scale_ + self.mu_
