import torch

from causal_nf.sem_equations.sem_base import SEM


class Collider(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda _, u2: 2.0 - u2,
                lambda x1, x2, u3: 0.25 * x2 - 0.5 * x1 + 0.5 * u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda _, x2: 2.0 - x2,
                lambda x1, x2, x3: (x3 - 0.25 * x2 + 0.5 * x1) / 0.5,
            ]
        elif sem_name == "non-linear":
            raise NotImplementedError
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((3, 3))

        adj[0, :] = torch.tensor([0, 0, 0])
        adj[1, :] = torch.tensor([0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0])
        if add_diag:
            adj += torch.eye(3)

        return adj

    def intervention_index_list(self):
        return [1]
