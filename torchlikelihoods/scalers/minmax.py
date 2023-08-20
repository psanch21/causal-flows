import torch
import torch.nn as nn

from .base import BaseScaler


class MinMaxScaler(BaseScaler):
    def __init__(self, feature_range=(0, 1)):
        self.min_ = float(feature_range[0])
        self.max_ = float(feature_range[1])

        self.feature_range = feature_range
        self.min_data = None
        self.max_data = None

        self.diff = self.max_ - self.min_

    def reset(self):
        self.min_data = None
        self.max_data = None

    def non_linearity(self):
        if self.feature_range == (0, 1):
            return nn.Sigmoid()
        elif self.feature_range == (-1, 1):
            return nn.Tanh()
        else:
            raise NotImplementedError

    def _fit_params(self, x, dims=None):
        assert isinstance(x, torch.Tensor)

        if isinstance(dims, tuple):
            min_data = x.min(dims[0], keepdim=True)[0]
            max_data = x.max(dims[0], keepdim=True)[0]
            for dim in dims[1:]:
                min_data = min_data.min(dim, keepdim=True)[0]
                max_data = max_data.max(dim, keepdim=True)[0]
        else:
            min_data = x.min()
            max_data = x.max()

        return {"min_data": min_data, "max_data": max_data}

    def aggregate_param(self, name, param):
        if name == "max_data":
            param_new = torch.max(param, dim=0)[0]
        elif name == "min_data":
            param_new = torch.min(param, dim=0)[0]
        else:
            raise NotImplementedError

        return param_new

    def fit_manual(self):
        self.min_data = self.min_
        self.max_data = self.max_

    def _transform(self, x, inplace):
        diff_data = self.max_data - self.min_data
        x_norm = (x - self.min_data) / diff_data * self.diff + self.min_
        return x_norm

    def _inverse_transform(self, x_norm, inplace):
        diff = self.max_data - self.min_data
        x = (x_norm - self.min_) * diff / self.diff + self.min_data
        return x


class MinMaxScaler_01(MinMaxScaler):
    def __init__(self):
        super(MinMaxScaler_01, self).__init__(feature_range=(0, 1))


class MinMaxScaler_n11(MinMaxScaler):
    def __init__(self):
        super(MinMaxScaler_n11, self).__init__(feature_range=(-1, 1))
