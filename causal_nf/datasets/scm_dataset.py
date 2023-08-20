import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset

from causal_nf.distributions import SCM

from causal_nf.sem_equations import sem_dict

from torch_geometric.utils import dense_to_sparse
import torch
from torch_geometric.data.data import Data


# %%
class SCMDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_samples: int,
        scm: SCM,
        name: str,
        sem_name: str,
        type: str = "torch",
        use_edge_attr: bool = False,
        seed: int = None,
    ):

        self.root_dir = root_dir
        self.name = name
        self.sem_name = sem_name

        sem_fn = sem_dict[name](sem_name=sem_name)
        self.adjacency = sem_fn.adjacency(True)
        self.num_nodes = self.adjacency.shape[0]
        self.type = type
        self.use_edge_attr = use_edge_attr
        self.seed = seed

        self.num_samples = num_samples

        self.scm = scm

        self.X = None
        self.U = None

    def _create_data(self):
        """
        This method sets the value for self.X and self.U
        Returns: None

        """
        U = self.scm.base_dist.sample((self.num_samples,))
        X = self.scm.transform(U)

        return X, U

    def prepare_data(self) -> None:
        print(f"\nPreparing data...")
        X, U = self._create_data()

        self.X = X
        self.U = U

        if self.type == "pyg":  # pytorch geometric
            edge_index = dense_to_sparse(self.adjacency)[0]
            self.edge_index = edge_index
            self._edge_attr = torch.eye(edge_index.shape[-1])
            self.node_ids = torch.eye(self.num_nodes)

    @property
    def edge_attr(self):
        if self.use_edge_attr:
            return self._edge_attr
        else:
            return None

    def __getitem__(self, index):
        if self.type == "torch":
            return self.X[index], self.U[index]
        elif self.type == "pyg":
            attr_dict = {}
            attr_dict["x"] = self.X[index].reshape(-1, 1)
            attr_dict["edge_index"] = self.edge_index
            attr_dict["node_ids"] = self.node_ids
            attr_dict["edge_attr"] = self.edge_attr
            return Data(**attr_dict)

    def __len__(self):
        return len(self.X)

    def __str__(self):
        my_str = f"Dataset {self.name}\n"
        my_str += f"\tnum_samples: {self.num_samples}\n"
        my_str += f"\tsem_name: {self.sem_name}\n"
        return my_str

    def plot(self, x, col_names=None, title=None):
        if col_names is None:
            col_names = [f"x_{i + 1}" for i in range(x.shape[1])]
        df = pd.DataFrame(x.numpy(), columns=col_names)
        g = sns.PairGrid(df, diag_sharey=False)

        g.map_upper(sns.scatterplot, s=15)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=2)
        g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp

        if title is None:
            g.fig.suptitle(f"Dataset: {self.name} {self.sem_name}")
        else:
            g.fig.suptitle(title)
        return g
