from abc import ABC, abstractmethod
import torch


class BaseScaler(ABC):
    def reset(self):
        return

    def to(self, device):
        for attr_name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr_name, value.to(device))

    @abstractmethod
    def _fit_params(self, x, dims=None):
        pass

    @abstractmethod
    def aggregate_param(self, name, param):
        pass

    @abstractmethod
    def non_linearity(self):
        raise NotImplementedError

    def fit(self, x, dims=None):
        if isinstance(x, list):
            self._fit_with_list(x, dims)
        else:
            self._fit(x, dims)

    def _fit(self, x, dims):
        params_dict = self._fit_params(x, dims)
        for key, value in params_dict.items():

            setattr(self, key, value)

    def _fit_with_list(self, x_list, dims=None):
        params_dict = {}
        for x in x_list:
            params_dict_i = self._fit_params(x, dims=None)
            for key, value in params_dict_i.items():
                if key not in params_dict:
                    params_dict[key] = []
                params_dict[key].append(value)

        for key, values in params_dict.items():
            value = self.aggregate_param(key, torch.stack(values))
            setattr(self, key, value)

    def fit_with_loader(self, loader, dims=None):
        i = 0
        x_list = []
        for batch in iter(loader):
            i += 1
            x_list.append(batch[0])
            if i == 60:
                print("Stopping fitting at batch 60")
                break

        x = torch.cat(x_list)

        self.fit(x, dims)

    @abstractmethod
    def fit_manual(self):
        pass

    def transform(self, x, inplace=True):
        if isinstance(x, list):
            x_norm = []
            for x_i in x:
                x_i_norm = self._transform(x_i, inplace)
                x_norm.append(x_i_norm)
            return x_norm
        else:
            return self._transform(x, inplace)

    @abstractmethod
    def _transform(self, x, inplace):
        pass

    def inverse_transform(self, x_norm, inplace=True):
        if isinstance(x_norm, list):
            x = []
            for x_i_norm in x_norm:
                x_i = self._inverse_transform(x_i_norm, inplace)
                x.append(x_i)
            return x
        else:
            return self._inverse_transform(x_norm, inplace)

    @abstractmethod
    def _inverse_transform(self, x_norm, inplace):
        pass

    def __str__(self):

        my_attr = []

        for key, value in self.__dict__.items():
            if value is not None:
                my_attr.append(f"\n\t\t{key}={value}")
            else:
                my_attr.append(f"\n\t\t{key}=None")

        my_str = f"{self.__class__.__name__}("
        my_str += ", ".join(my_attr) + "\n)"
        return my_str
