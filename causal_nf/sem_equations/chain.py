import torch

from causal_nf.sem_equations.sem_base import SEM


class Chain(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 10 * x1 - u2,
                lambda x1, x2, u3: 0.25 * x2 + 2 * u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (10 * x1 - x2),
                lambda x1, x2, x3: (x3 - 0.25 * x2) / 2,
            ]
        elif sem_name == "non-linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.exp(x1 / 2.0) + u2 / 4.0,
                lambda x1, x2, u3: (x2 - 5) ** 3 / 15.0 + u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: 4.0 * (x2 - torch.exp(x1 / 2.0)),
                lambda x1, x2, x3: x3 - (x2 - 5) ** 3 / 15.0,
            ]
        elif sem_name == "non-linear-2":
            functions = [
                lambda u1: torch.sigmoid(u1),
                lambda x1, u2: 10 * x1**0.5 - u2,
                lambda x1, x2, u3: 0.25 * x2 + 2 * u3,
            ]
            inverses = [
                lambda x1: torch.logit(x1),
                lambda x1, x2: (10 * x1**0.5 - x2),
                lambda x1, x2, x3: (x3 - 0.25 * x2) / 2,
            ]

        elif sem_name == "non-linear-3":

            functions = [
                lambda u1: u1,
                lambda x1, u2: 1 * x1**2.0 - u2,
                lambda x1, x2, u3: 0.25 * x2 + 2 * u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (1 * x1**2.0 - x2),
                lambda x1, x2, x3: (x3 - 0.25 * x2) / 2,
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((3, 3))

        adj[0, :] = torch.tensor([0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0])
        adj[2, :] = torch.tensor([0, 1, 0])
        if add_diag:
            adj += torch.eye(3)

        return adj

    def intervention_index_list(self):
        return [0, 1]
