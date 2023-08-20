import torch.distributions as td
from torch.utils.data import DataLoader

from .base import BaseLikelihood
from ..scalers import StandardScaler


class NormalMeanLikelihood(BaseLikelihood):
    def __init__(self, domain_size, std):
        super().__init__(domain_size)

        self.std = std

    def _params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):
        mu = logits
        p = td.Normal(mu, self.std)
        if return_mean:
            return p.mean, p
        else:
            return p

    def _get_scaler(self):
        return StandardScaler()


class NormalMean1Likelihood(NormalMeanLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size, std=1)


class NormalMean01Likelihood(NormalMeanLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size, std=0.1)
