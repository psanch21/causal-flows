import torch

from .base import BaseScaler


class HeterogeneousScaler(BaseScaler):
    def __init__(self, scalers, splits):

        self.scalers = scalers
        self.splits = splits

    def non_linearity(self):
        non_linearity = []
        for scaler in self.scalers:
            non_linearity_i = scaler.non_linearity()
            non_linearity.append(non_linearity_i)
        return non_linearity

    def _get_x_list(self, x):
        x_list = torch.split(x, split_size_or_sections=self.splits, dim=1)
        return x_list

    def to(self, device):
        for scaler in self.scalers:
            scaler.to(device)

    def _fit_params(self, x, dims=None):
        raise NotImplementedError

    def aggregate_param(self, name, param):
        raise NotImplementedError

    def _fit(self, x, dims=None):
        x_list = self._get_x_list(x)

        for x_i, scaler in zip(x_list, self.scalers):
            scaler.fit(x_i, dims=dims)

    def _fit_with_list(self, x_list, dims=None):
        x_i_dict = {}
        for x in x_list:
            x_i_list = self._get_x_list(x)
            for i, x_i in enumerate(x_i_list):
                if i not in x_i_dict:
                    x_i_dict[i] = []
                x_i_dict[i].append(x_i)

        for i, x_i_list in x_i_dict.items():
            self.scalers[i]._fit_with_list(x_i_list)

    def fit_manual(self):
        for scaler in self.scalers:
            scaler.fit_manual()

    def _transform(self, x, inplace):
        x_list = self._get_x_list(x)
        x_norm_list = []
        for x_i, scaler in zip(x_list, self.scalers):
            x_norm_i = scaler.transform(x_i)
            x_norm_list.append(x_norm_i)

        x_norm = torch.cat(x_norm_list, dim=-1)

        return x_norm

    def _inverse_transform(self, x_norm, inplace):
        x_norm_list = self._get_x_list(x_norm)
        x_list = []
        for x_norm_i, scaler in zip(x_norm_list, self.scalers):
            x_i = scaler.inverse_transform(x_norm_i)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=-1)

        return x
