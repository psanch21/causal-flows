import pytest
import torch
import torch.distributions as distr

from causal_nf.distributions.heterogeneous import Heterogeneous
from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import CausalEquations

from causal_nf.preparators.scm._base_distributions import pu_dict

from causal_nf.distributions.scm import SCM

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from tueplots import bundles

plt.rcParams.update(bundles.icml2022())


def assert_scm(p_u, num_samples, dim, sem, intervention_idx, intervention_value):
    u = p_u.sample((num_samples,))
    assert u.shape == (num_samples, dim)

    x = sem(u)
    u_hat = sem.inv(x)
    for i in range(num_samples):
        assert torch.allclose(
            u[i], u_hat[i], rtol=5e-2
        ), f"sample[{i}]: {u[i]} {u_hat[i]}"

    scm = SCM(base_distribution=p_u, transform=sem)

    log_prob = scm.log_prob(x)
    assert log_prob.shape == (num_samples,)

    scm.intervene(index=intervention_idx, value=intervention_value)
    x_cf = scm.sample_counterfactual(factual=x)
    value = torch.ones((num_samples,)) * intervention_value
    assert torch.allclose(x_cf[:, intervention_idx], value, rtol=1e-2)

    x_int = sem(u)
    assert torch.allclose(x_int[:, intervention_idx], value, rtol=1e-2)
    sem.stop_intervening(index=intervention_idx)

    ate = scm.compute_ate(index=intervention_idx, a=0, b=3, num_samples=10000)

    assert len(ate) == (dim - 1)


@pytest.mark.parametrize(
    "name", ["chain", "chain-4", "chain-5", "collider", "fork", "triangle"]
)
@pytest.mark.parametrize("sem_name", ["linear"])
@pytest.mark.parametrize("num_samples", [1, 100])
@pytest.mark.parametrize("intervention_idx", [0, 1, 2])
@pytest.mark.parametrize("intervention_value", [-1.0, -0.1, 0.0, 0.1, 1.0])
@pytest.mark.parametrize("base_version", [1, 2, 3])
def test_scm_linear(
    name, sem_name, num_samples, intervention_idx, intervention_value, base_version
):
    sem_fn = sem_dict[name](sem_name=sem_name)
    assert isinstance(sem_fn.intervention_index_list(), list)

    sem = CausalEquations(
        functions=sem_fn.functions, inverses=sem_fn.inverses, derivatives=None
    )

    dim = len(sem_fn.functions)

    p_u = pu_dict[dim](version=base_version)

    assert_scm(p_u, num_samples, dim, sem, intervention_idx, intervention_value)


@pytest.mark.parametrize(
    "name", ["chain", "triangle", "diamond", "fork", "simpson", "large-backdoor"]
)
@pytest.mark.parametrize("sem_name", ["non-linear"])
@pytest.mark.parametrize("num_samples", [1, 100])
@pytest.mark.parametrize("intervention_idx", [0, 1, 2])
@pytest.mark.parametrize("intervention_value", [-1.0, -0.1, 0.0, 0.1, 1.0])
@pytest.mark.parametrize("base_version", [1, 2, 3])
def test_scm_non_linear(
    name, sem_name, num_samples, intervention_idx, intervention_value, base_version
):
    sem_fn = sem_dict[name](sem_name=sem_name)
    assert isinstance(sem_fn.intervention_index_list(), list)

    sem = CausalEquations(
        functions=sem_fn.functions, inverses=sem_fn.inverses, derivatives=None
    )

    dim = len(sem_fn.functions)

    if dim > 5:
        base_version = 1

    p_u = pu_dict[dim](version=base_version)

    assert_scm(p_u, num_samples, dim, sem, intervention_idx, intervention_value)


@pytest.mark.parametrize("name", ["simpson"])
@pytest.mark.parametrize("sem_name", ["sym-prod"])
@pytest.mark.parametrize("num_samples", [1, 100])
@pytest.mark.parametrize("intervention_idx", [0, 1, 2])
@pytest.mark.parametrize("intervention_value", [-1.0, -0.1, 0.0, 0.1, 1.0])
@pytest.mark.parametrize("base_version", [1, 2, 3])
def test_scm_sym_prod(
    name, sem_name, num_samples, intervention_idx, intervention_value, base_version
):
    sem_fn = sem_dict[name](sem_name=sem_name)

    assert isinstance(sem_fn.intervention_index_list(), list)

    sem = CausalEquations(
        functions=sem_fn.functions, inverses=sem_fn.inverses, derivatives=None
    )

    dim = len(sem_fn.functions)

    p_u = pu_dict[dim](version=base_version)

    assert_scm(p_u, num_samples, dim, sem, intervention_idx, intervention_value)
