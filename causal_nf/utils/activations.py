import torch
import torch.nn as nn


def sin_activation(input):
    return torch.sin(input)


class SinActivation(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return sin_activation(input)  # simply apply already implemented SiLU


def get_act_fn(activation):
    if activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "selu":
        return nn.SELU(inplace=False)
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "elu":
        return nn.ELU(inplace=False)
    if "lrelu" in activation:
        return nn.LeakyReLU(
            negative_slope=float(activation.split("__")[1].replace("_", ".")),
            inplace=False,
        )
    elif activation == "softmax":
        return nn.Softmax()
    elif activation == "identity":
        return nn.Identity()
    elif activation == "sinus":
        return SinActivation()
    else:
        raise NotImplementedError
