import torch
import torch.nn as nn


class DisjointDense(nn.Module):
    def __init__(self, in_dimension: int, out_dimension: int, num_disjoint: int):
        super(DisjointDense, self).__init__()

        self.weights = nn.Linear(num_disjoint, in_dimension * out_dimension, bias=False)

        self.bias = nn.Linear(num_disjoint, out_dimension, bias=False)

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

    def reset_parameters(self):
        self.weights.reset_parameters()
        self.bias.reset_parameters()

    def forward(self, x, one_hot_selector):
        w = self.weights(one_hot_selector).view(
            -1, self.in_dimension, self.out_dimension
        )  # [N, in, out]
        h = torch.einsum("bij,bi->bj", w, x)  # [N, out]

        bias = self.bias(one_hot_selector)

        return h + bias
