import torch

from causal_nf.sem_equations.sem_base import SEM


class Fork(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda _, u2: 2.0 - u2,
                lambda x1, x2, u3: 0.25 * x2 - 1.5 * x1 + 0.5 * u3,
                lambda _, __, x3, u4: 1.0 * x3 + 0.25 * u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda _, x2: 2.0 - x2,
                lambda x1, x2, x3: (x3 - 0.25 * x2 + 1.5 * x1) / 0.5,
                lambda _, __, x3, x4: (x4 - 1.0 * x3) / 0.25,
            ]
        elif sem_name == "non-linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: u2,
                lambda x1, x2, u3: 4.0 / (1 + torch.exp(-x1 - x2)) - x2**2 + u3 / 2.0,
                lambda x1, x2, x3, u4: 20.0 / (1 + torch.exp(x3**2 / 2 - x3)) + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2,
                lambda x1, x2, x3: 2.0
                * (x3 - 4.0 / (1 + torch.exp(-x1 - x2)) + x2**2),
                lambda x1, x2, x3, x4: x4 - 20.0 / (1 + torch.exp(x3**2 / 2 - x3)),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([0, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0])
        adj[3, :] = torch.tensor([0, 0, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [1, 2]
