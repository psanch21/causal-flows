from typing import List

import torch
import torch.distributions as td

from ..likelihoods import CategoricalLikelihood
from ..likelihoods.base import BaseLikelihood

flatten = lambda t: [item for sublist in t for item in sublist]


class HeterogeneousDistribution:
    def __init__(
        self,
        likelihoods: List[BaseLikelihood],
        norm_categorical: bool,  # True, False
        norm_by_dim: int,  # Unsigned Integer
    ):
        assert isinstance(likelihoods, list)

        self.norm_categorical = norm_categorical

        self.likelihoods = likelihoods
        self.params_size_list = []
        self._domain_size_index_list = []
        self._domain_size_one_hot_list = []
        self.norm_by_dim = norm_by_dim
        for lik in likelihoods:
            self.params_size_list.append(lik.params_size(flatten=True))
            self._domain_size_one_hot_list.append(lik.domain_size())
            if isinstance(lik, CategoricalLikelihood):
                self._domain_size_index_list.append(1)
            else:
                self._domain_size_index_list.append(lik.domain_size())

        self.params_size = sum(self.params_size_list)
        self._domain_size_index = sum(self._domain_size_index_list)
        self._domain_size_one_hot = sum(self._domain_size_one_hot_list)

    def domain_size(self, one_hot=True):
        if one_hot:
            return self._domain_size_one_hot
        else:
            return self._domain_size_index

    def domain_size_list(self, one_hot=True):
        if one_hot:
            return self._domain_size_one_hot_list
        else:
            return self._domain_size_index_list

    def set_logits(self, logits):
        self.distributions = []
        logits_list = torch.split(
            logits, split_size_or_sections=self.params_size_list, dim=1
        )
        for lik_i, logits_i in zip(self.likelihoods, logits_list):
            self.distributions.append(lik_i(logits_i))

    @property
    def mean(self):
        means = []
        for i, distr in enumerate(self.distributions):
            if isinstance(distr, td.Categorical):
                means.append(distr.probs)
            else:
                means.append(distr.mean)
        return torch.cat(means, dim=1)

    def sample(self, sample_shape=torch.Size(), one_hot=True):
        samples = []
        for i, distr in enumerate(self.distributions):
            sample_i = distr.sample(sample_shape)
            print(f"\n{distr}")

            if one_hot and isinstance(distr, td.Categorical):
                y_onehot = torch.FloatTensor(distr.probs.shape)
                # In your for loop
                y_onehot.zero_()
                y_onehot.scatter_(1, sample_i.view(-1, 1), 1)
                sample_i = y_onehot

            if sample_i.ndim == 1:
                sample_i = sample_i.unsqueeze(-1)
            samples.append(sample_i)

        return torch.cat(samples, dim=1)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def log_prob(self, x):
        domain_one_hot = True
        if x.shape[1] == self.domain_size(True):
            value_list = torch.split(
                x, split_size_or_sections=self.domain_size_list(True), dim=1
            )
        elif x.shape[1] == self.domain_size(False):
            domain_one_hot = False
            value_list = torch.split(
                x, split_size_or_sections=self.domain_size_list(False), dim=1
            )
        else:
            raise Exception("Wrong dimension of x")

        log_probs = []
        for i, (value_i, distr_i) in enumerate(zip(value_list, self.distributions)):
            if isinstance(distr_i, td.Categorical):
                if domain_one_hot:
                    num_categories = value_i.shape[1]
                    value_i = value_i.argmax(-1)
                else:
                    num_categories = self.domain_size_list(True)[i]
                    value_i = value_i.flatten()

                log_prob_i = distr_i.log_prob(value_i).unsqueeze(-1)
                if self.norm_categorical:
                    log_prob_i = log_prob_i / num_categories
                log_probs.append(log_prob_i)
            else:
                log_prob_i = distr_i.log_prob(value_i)
                if self.norm_by_dim == 1:  # Normalize by dimension
                    log_prob_i = log_prob_i / log_prob_i.shape[1]
                elif self.norm_by_dim > 1:
                    log_prob_i = log_prob_i / self.norm_by_dim
                log_probs.append(log_prob_i)

        return torch.cat(log_probs, dim=1)
