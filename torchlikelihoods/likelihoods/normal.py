import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseLikelihood

from ..scalers import StandardScaler


class NormalLikelihood(BaseLikelihood):
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
        assert (
            logits.shape[self.dim + 1] == self.params_size()
        ), f"Logits shape {logits.shape[self.dim]} does not match params size {self.params_size()}"
        latent_dim = logits.size(self.dim + 1) // 2
        mu, log_var = torch.split(
            logits, split_size_or_sections=latent_dim, dim=self.dim + 1
        )
        log_var = torch.clamp(log_var, min=-70, max=70)
        std = torch.exp(log_var / 2)
        std = torch.clamp(std, min=0.001, max=10)

        p = td.Normal(mu, std)
        if return_mean:
            return mu, p
        else:
            return p

    def _get_scaler(self):
        return StandardScaler()
