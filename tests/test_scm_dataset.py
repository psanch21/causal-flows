import os

import pytest
import torch_geometric.data as pygdata
from torch.utils.data import DataLoader

from causal_nf.datasets.scm_dataset import SCMDataset
from causal_nf.distributions.scm import SCM
from causal_nf.preparators.scm._base_distributions import pu_dict
from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import CausalEquations

@pytest.mark.parametrize("name", ["chain", "triangle"])
@pytest.mark.parametrize("sem_name", ["non-linear"])
@pytest.mark.parametrize("num_samples", [1, 128, 1024])
def test_scm_dataset(name, sem_name, num_samples):
    root_dir = os.path.join("..", "Data")

    sem_fn = sem_dict[name](sem_name=sem_name)

    sem = CausalEquations(
        functions=sem_fn.functions, inverses=sem_fn.inverses, derivatives=None
    )

    p_u = pu_dict[3](version=1)

    scm = SCM(base_distribution=p_u, transform=sem)

    dataset = SCMDataset(
        root_dir=root_dir,
        num_samples=num_samples,
        scm=scm,
        name=name,
        sem_name=sem_name,
    )

    dataset.prepare_data()

    assert len(dataset) == num_samples
    assert dataset.X.shape == (num_samples, 3)
    assert dataset.U.shape == (num_samples, 3)
    batch_size = min(num_samples, 128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x, u = next(iter(loader))
    assert x.shape == (batch_size, 3)
    assert u.shape == (batch_size, 3)


@pytest.mark.parametrize("name", ["chain", "triangle"])
@pytest.mark.parametrize("sem_name", ["non-linear"])
@pytest.mark.parametrize("num_samples", [1, 128, 1024])
def test_scm_dataset_pyg(name, sem_name, num_samples):
    root_dir = os.path.join("..", "Data")

    sem_fn = sem_dict[name](sem_name=sem_name)

    sem = CausalEquations(
        functions=sem_fn.functions, inverses=sem_fn.inverses, derivatives=None
    )

    p_u = pu_dict[3](version=1)

    scm = SCM(base_distribution=p_u, transform=sem)

    dataset = SCMDataset(
        root_dir=root_dir,
        num_samples=num_samples,
        scm=scm,
        name=name,
        type="pyg",
        sem_name=sem_name,
    )

    dataset.prepare_data()

    assert len(dataset) == num_samples
    assert dataset.X.shape == (num_samples, 3)
    assert dataset.U.shape == (num_samples, 3)
    batch_size = min(num_samples, 128)
    loader = pygdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batch = next(iter(loader))
    assert batch.x.reshape(batch_size, -1).shape == (batch_size, 3)
