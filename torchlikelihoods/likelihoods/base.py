import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import torch


class BaseLikelihood(nn.Module):
    def __init__(self, domain_size, num_workers=0):

        self.num_workers = num_workers
        self.dim = 0

        self.is_list = isinstance(domain_size, list)
        self._domain_size = domain_size
        super(BaseLikelihood, self).__init__()

    def domain_size(self, flatten=True):
        if flatten and self.is_list:
            return int(np.prod(self._domain_size))
        else:
            return self._domain_size

    def params_size(self, flatten=True):
        params_size = self._params_size()
        if flatten and self.is_list:
            return int(np.prod(params_size))
        else:
            return params_size

    def _params_size(self):
        raise NotImplementedError()

    def _get_scaler(self):
        raise NotImplementedError()

    def get_scaler(self, dataset, dims=None):

        if dataset is None:
            scaler = self._get_scaler()
            scaler.fit_manual(dims=dims)
            return scaler
        elif isinstance(dataset, Dataset):
            bs = 64
            loader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            scaler = self._get_scaler()
            self.fit_scaler_with_loader(scaler, loader, dims=dims)

            return scaler
        elif isinstance(dataset, torch.Tensor):
            scaler = self._get_scaler()
            self.fit_scaler(scaler, dataset, dims=dims)

            return scaler

    def fit_scaler(self, scaler, x, dims=None):
        scaler.fit(x, dims=dims)

    def fit_scaler_with_loader(self, scaler, loader, dims=None):
        print("Fitting scaler with loader")
        scaler.fit_with_loader(loader, dims=dims)
