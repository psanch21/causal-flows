import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseLikelihood

from ..scalers import MinMaxScaler


class BernoulliLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    def _params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):
        p = td.Bernoulli(logits=logits)
        if return_mean:
            return p.mean, p
        else:
            return p

    def _get_scaler(self):
        return MinMaxScaler(feature_range=(0, 1))
