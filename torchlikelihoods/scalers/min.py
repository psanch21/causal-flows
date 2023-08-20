from .base import BaseScaler
import torch

import torch.nn as nn


class MinScaler(BaseScaler):
    def __init__(self, min_=0):
        assert min_ == 0

        self.min_ = min_
        self.min_data = None

    def non_linearity(self):
        return nn.ReLU()

    def reset(self):
        self.min_data = None

    def _fit_params(self, x, dims=None):
        assert isinstance(x, torch.Tensor)

        if isinstance(dims, tuple):
            min_data = x.min(dims[0], keepdim=True)[0]
            for dim in dims[1:]:
                min_data = min_data.min(dim, keepdim=True)[0]
        else:
            min_data = x.min()

        return {
            "min_data": min_data,
        }

    def aggregate_param(self, name, param):

        if name == "min_data":
            param_new = torch.min(param, dim=0)[0]
        else:
            raise NotImplementedError

        return param_new

    def fit_manual(self):
        self.min_data = 0.0

    def _transform(self, x, inplace):
        x_norm = x - self.min_data
        return x_norm

    def _inverse_transform(self, x_norm, inplace):
        x = x_norm + self.min_data
        return x
