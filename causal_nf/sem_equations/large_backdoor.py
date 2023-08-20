import torch
import torch.nn.functional as F

from causal_nf.sem_equations.sem_base import SEM


class LargeBackdoor(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None

        def inv_softplus(bias):
            return bias.expm1().clamp_min(1e-6).log()

        def layer(x, y):
            return F.softplus(x + 1) + F.softplus(0.5 + y) - 3.0

        def inv_layer(x, z):
            return inv_softplus(z + 3 - F.softplus(x + 1)) - 0.5

        def icdf_laplace(loc, scale, value):
            term = value - 0.5
            return loc - scale * term.sign() * torch.log1p(-2 * term.abs())

        def cdf_laplace(loc, scale, value):
            return 0.5 - 0.5 * (value - loc).sign() * torch.expm1(
                -(value - loc).abs() / scale
            )

        if sem_name == "non-linear":
            functions = [
                lambda u1: F.softplus(1.8 * u1) - 1,  # x1
                lambda x1, u2: 0.25 * u2 + layer(x1, torch.zeros_like(u2)) * 1.5,  # x2
                lambda x1, x2, u3: layer(x1, u3),  # x3
                lambda *args: layer(args[1], args[-1]),  # x4
                lambda *args: layer(args[2], args[-1]),  # x5
                lambda *args: layer(args[3], args[-1]),  # x6
                lambda *args: layer(args[4], args[-1]),  # x7
                lambda *args: 0.3 * args[-1] + (F.softplus(args[5] + 1) - 1),  # x8
                lambda *args: icdf_laplace(
                    -F.softplus((args[6] * 1.3 + args[7]) / 3 + 1) + 2, 0.6, args[-1]
                ),  # x9
            ]
            inverses = [
                lambda *args: inv_softplus(args[-1] + 1) / 1.8,  # u1
                lambda *args: 4
                * (-layer(args[0], torch.zeros_like(args[-1])) * 1.5 + args[-1]),  # u2
                lambda *args: inv_layer(args[0], args[-1]),  # u3
                lambda *args: inv_layer(args[1], args[-1]),  # u4
                lambda *args: inv_layer(args[2], args[-1]),  # u5
                lambda *args: inv_layer(args[3], args[-1]),  # u6
                lambda *args: inv_layer(args[4], args[-1]),  # u7
                lambda *args: (args[-1] - F.softplus(args[5] + 1) + 1) / 0.3,  # u8
                lambda *args: cdf_laplace(
                    -F.softplus((args[6] * 1.3 + args[7]) / 3 + 1) + 2, 0.6, args[-1]
                ),  # u9
            ]

        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
            ]
        ).float()

        if add_diag:
            adj += torch.eye(9)

        return adj

    def intervention_index_list(self):
        return [0, 1, 2, 4]
