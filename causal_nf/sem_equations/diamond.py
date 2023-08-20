import torch

from causal_nf.sem_equations.sem_base import SEM


class Diamond(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = None
            inverses = None
        elif sem_name == "non-linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1**2 + u2 / 2,
                lambda x1, x2, u3: x2**2 - 2.0 / (1 + torch.exp(-x1)) + u3 / 2.0,
                lambda x1, x2, x3, u4: x3 / ((x2 + 2.0).abs() + x3 + 0.5) + u4 / 10.0,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: 2 * (x2 - x1**2),
                lambda x1, x2, x3: (x3 - x2**2 + 2.0 / (1 + torch.exp(-x1))) * 2.0,
                lambda x1, x2, x3, x4: 10 * (x4 - x3 / ((x2 + 2.0).abs() + x3 + 0.5)),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0])
        adj[3, :] = torch.tensor([0, 1, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [0, 1, 2]
