import torch
import torch.nn.functional as F

from causal_nf.sem_equations.sem_base import SEM


class GermanCredit(SEM):
    def __init__(self, sem_name="dummy"):
        functions = None
        inverses = None

        if sem_name == "dummy":
            functions = [
                lambda *args: args[-1],  # x1
                lambda *args: args[-1],  # x2
                lambda *args: args[-1],  # x3
                lambda *args: args[-1],  # x4
                lambda *args: args[-1],  # x5
                lambda *args: args[-1],  # x6
                lambda *args: args[-1],  # x7
            ]
            inverses = [
                lambda *args: args[-1],  # u1
                lambda *args: args[-1],  # u2
                lambda *args: args[-1],  # u3
                lambda *args: args[-1],  # u4
                lambda *args: args[-1],  # u5
                lambda *args: args[-1],  # u6
                lambda *args: args[-1],  # u7
            ]

        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((7, 7))
        adj[0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        adj[1, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0, 0, 0, 0])
        adj[3, :] = torch.tensor([1, 1, 1, 0, 0, 0, 0])
        adj[4, :] = torch.tensor([1, 1, 0, 0, 0, 0, 0])
        adj[5, :] = torch.tensor([1, 1, 0, 0, 1, 0, 0])
        adj[6, :] = torch.tensor([1, 1, 0, 0, 1, 1, 0])

        if add_diag:
            adj += torch.eye(7)

        return adj

    def intervention_index_list(self):
        return [0, 1]
