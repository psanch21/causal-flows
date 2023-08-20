import torch.distributions as td
from torch.utils.data import DataLoader

from .base import BaseLikelihood
from ..scalers import StandardScaler


from ..distributions import Delta


class DeltaLikelihood(BaseLikelihood):
    lambda_ = None

    def __init__(self, domain_size):
        super().__init__(domain_size)

        self.lambda_ = DeltaLikelihood.lambda_

    # a class method to create a Person object by birth year.
    @classmethod
    def create(cls, lambda_):
        cls.lambda_ = lambda_
        return cls

    def _params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):
        mu = logits
        p = Delta(center=logits, lambda_=self.lambda_)
        if return_mean:
            return p.mean, p
        else:
            return p

    def _get_scaler(self):
        return StandardScaler()
