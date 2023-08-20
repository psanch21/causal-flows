import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseLikelihood

from ..scalers import MinMaxScaler


class BetaLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    def _params_size(self):
        if self.is_list:
            params_size = self._domain_size.copy()
            params_size[self.dim] = params_size[self.dim] * 2
            return params_size
        else:
            return self._domain_size * 2

    def forward(self, logits, return_mean=False):
        logits = F.softplus(logits)
        latent_dim = logits.size(self.dim + 1) // 2
        c0, c1 = torch.split(
            logits, split_size_or_sections=latent_dim, dim=self.dim + 1
        )
        p = td.Beta(c0, c1)
        if return_mean:
            return p.mean, p
        else:
            return p

    def _get_scaler(self):
        return MinMaxScaler(feature_range=(0, 1))
