from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter


# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py


def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce="sum")


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce="mean")


def aggregate_mul(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce="mul")


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce="min")


def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce="max")


def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


AGGREGATORS = {
    "sum": aggregate_sum,
    "mean": aggregate_mean,
    "min": aggregate_min,
    "max": aggregate_max,
    "var": aggregate_var,
    "std": aggregate_std,
}

from typing import Dict

import torch
from torch import Tensor


# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py


def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg["log"])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg["log"] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg["lin"])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg["lin"] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
    "linear": scale_linear,
    "inverse_linear": scale_inverse_linear,
}
