import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseLikelihood

from ..distributions import HeterogeneousDistribution

from ..scalers import HeterogeneousScaler
from typing import List, Any, Dict, Tuple


class HeterogeneousLikelihood(BaseLikelihood):
    def __init__(
        self,
        likelihoods: List[BaseLikelihood],
        norm_categorical: bool,
        norm_by_dim: bool,
        one_hot_domain=True,
    ):

        assert one_hot_domain

        self.distr = HeterogeneousDistribution(
            likelihoods=likelihoods,
            norm_categorical=norm_categorical,
            norm_by_dim=norm_by_dim,
        )

        super().__init__(self.distr.domain_size(one_hot_domain))

    def _params_size(self):
        return self.distr.params_size

    def forward(self, logits, return_mean=False):
        self.distr.set_logits(logits)
        if return_mean:
            return self.distr.mean, self.distr
        else:
            return self.distr

    def _get_scaler(self):
        scalers = [lik._get_scaler() for lik in self.distr.likelihoods]
        splits = [lik.domain_size() for lik in self.distr.likelihoods]
        return HeterogeneousScaler(scalers=scalers, splits=splits)
