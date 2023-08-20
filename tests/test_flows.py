import pytest
import torch
import torch.distributions as distr

from causal_nf.distributions.heterogeneous import Heterogeneous
from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import CausalEquations

from causal_nf.preparators.scm._base_distributions import pu_dict

from causal_nf.distributions.scm import SCM
import zuko.flows as zflows


@pytest.mark.parametrize("flow_name", ["maf", "nsf"])
@pytest.mark.parametrize("num_samples", [1, 100])
@pytest.mark.parametrize("hidden_features", [1, 3, 64])
@pytest.mark.parametrize("base_to_data", [False, True])
@pytest.mark.parametrize("base_distr", ["normal", "laplace"])
@pytest.mark.parametrize("learn_base", [True, False])
def test_flows(
    flow_name, num_samples, hidden_features, base_to_data, base_distr, learn_base
):
    if flow_name == "maf":
        flow = zflows.MAF(
            features=3,
            context=0,
            transforms=3,
            hidden_features=[hidden_features] * 3,
            passes=None,
            base_to_data=base_to_data,
            base_distr=base_distr,
            learn_base=learn_base,
        )
    elif flow_name == "unaf":
        flow = zflows.UNAF(3, 0, transforms=3, hidden_features=[hidden_features] * 3)
    elif flow_name == "nsf":
        flow = zflows.NSF(
            3,
            0,
            transforms=3,
            hidden_features=[hidden_features] * 3,
            base_to_data=base_to_data,
            base_distr=base_distr,
            learn_base=learn_base,
        )
    elif flow_name == "naf":
        flow = zflows.NAF(
            features=3,
            context=0,
            transforms=3,
            hidden_features=[hidden_features] * 3,
            randperm=False,
        )

    n_flow = flow()
    x = n_flow.sample((num_samples,))
    assert x.shape == (num_samples, 3)

    log_prob = n_flow.log_prob(x)
    assert log_prob.shape == (num_samples,)
    u = n_flow.transform.inv(x)
    assert u.shape == (num_samples, 3)
    x_hat = n_flow.transform(u)
    for i in range(num_samples):
        assert torch.allclose(
            x[i], x_hat[i], rtol=1e-2
        ), f"sample[{i}]: {x[i]} {x_hat[i]}"
