import torch

from causal_nf.sem_equations.sem_base import SEM


class Chain5(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 10 * x1 - u2,
                lambda _1, x2, u3: 0.25 * x2 + 2 * u3,
                lambda _1, _2, x3, u4: x3 + u4,
                lambda _1, _2, _3, x4, u5: -x4 + u5,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (10 * x1 - x2),
                lambda _1, x2, x3: (x3 - 0.25 * x2) / 2,
                lambda _1, _2, x3, x4: x4 - x3,
                lambda _1, _2, _3, x4, x5: x5 + x4,
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        dim = len(self.inverses)
        adj = torch.zeros((dim, dim))

        adj[0, :] = torch.tensor([0, 0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0, 0])
        adj[2, :] = torch.tensor([0, 1, 0, 0, 0])
        adj[3, :] = torch.tensor([0, 0, 1, 0, 0])
        adj[4, :] = torch.tensor([0, 0, 0, 1, 0])
        if add_diag:
            adj += torch.eye(dim)

        return adj

    def intervention_index_list(self):
        return [0, 1, 2]
