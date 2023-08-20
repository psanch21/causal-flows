import copy

from .base import BaseScaler


class HeterogeneousObjectScaler(BaseScaler):
    def __init__(self, scalers_dict):

        self.scalers_dict = scalers_dict

    def non_linearity(self):
        non_linearity = []
        for attr_name, scaler in self.scalers_dict.items():
            non_linearity_i = scaler.non_linearity()
            non_linearity.append(non_linearity_i)
        return non_linearity

    def to(self, device):
        for attr_name, scaler in self.scalers_dict.items():
            scaler.to(device)

    def _fit_params(self, x, dims=None):
        raise NotImplementedError

    def aggregate_param(self, name, param):
        raise NotImplementedError

    def _fit(self, x, dims=None):

        for attr_name, scaler in self.scalers_dict.items():
            if isinstance(scaler, str):
                continue
            attr = getattr(x, attr_name)
            scaler.fit(attr, dims=dims)

    def _fit_with_list(self, x_list, dims=None):
        """List of objects"""
        attr_dict = {}
        for x in x_list:
            for attr_name, _ in self.scalers_dict.items():
                attr = getattr(x, attr_name)
                if attr_name not in attr_dict:
                    attr_dict[attr_name] = []
                attr_dict[attr_name].append(attr)

        for attr_name, scaler in self.scalers_dict.items():
            if isinstance(scaler, str):
                continue
            attr_list = attr_dict[attr_name]
            scaler._fit_with_list(attr_list)

    def fit_manual(self):
        for attr_name, scaler in self.scalers_dict.items():
            if isinstance(scaler, str):
                continue
            scaler.fit_manual()

    def _transform(self, x, inplace):
        if not inplace:
            x = copy.deepcopy(x)
        for attr_name, scaler in self.scalers_dict.items():
            attr = getattr(x, attr_name)
            if isinstance(scaler, str):
                scaler = self.scalers_dict[scaler]
            setattr(x, attr_name, scaler.transform(attr))
        return x

    def _inverse_transform(self, x_norm, inplace):
        if not inplace:
            x_norm = copy.deepcopy(x_norm)
        for attr_name, scaler in self.scalers_dict.items():
            attr_norm = getattr(x_norm, attr_name)
            if isinstance(scaler, str):
                scaler = self.scalers_dict[scaler]
            setattr(x_norm, attr_name, scaler.inverse_transform(attr_norm))

        return x_norm

    def __str__(self):
        my_str = ""
        for i, (attr_name, scaler) in enumerate(self.scalers_dict.items()):
            if isinstance(scaler, str):
                scaler = self.scalers_dict[scaler]
            my_str += f"---------- {i} ----------\n"
            my_str += f"{scaler}\n"

        return my_str
