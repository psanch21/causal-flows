import torch.nn as nn

from causal_nf.transforms import CausalTransform
import torch
from torch import Tensor

from torch import Tensor, Size

from zuko.transforms import ComposedTransform


class CausalNormalizingFlow(nn.Module):
    def __init__(self, flow):
        super(CausalNormalizingFlow, self).__init__()
        self.flow = flow
        self.adjacency = None


    def set_adjacency(self, adj):
        self.adjacency = adj

    def forward(self, x: Tensor) -> Tensor:
        output = {}
        n_flow = self.flow()

        log_prob = n_flow.log_prob(x)
        output["log_prob"] = log_prob

        output["loss"] = -log_prob
        return output

    def vi(self, x: Tensor, p, cte) -> Tensor:
        output = {}

        log_prob_p = p.log_prob(x)
        output["log_prob"] = log_prob_p

        n_flow = self.flow()
        log_prob_q = n_flow.log_prob(x)

        output["loss"] = log_prob_q - cte * log_prob_p
        return output

    def sample(self, shape):
        output = {}

        n_flow = self.flow()
        x, u = n_flow.sample_u(shape)

        output["u_obs"] = u
        output["x_obs"] = x

        return output

    @torch.no_grad()
    def compute_counterfactual_2(
        self, x_factual, index: int, value: float, scaler=None
    ) -> Tensor:

        n_flow = self.flow()
        if scaler is not None:  # x is not normalized
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        z_factual = n_flow.transform(x_factual)

        adj_int = self.adjacency.clone()
        adj_int[index, :] = 0.0
        self.flow.intervene(adj_int)

        n_flow = self.flow()
        if scaler is not None:  # x is not normalized
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        x_int = torch.zeros_like(x_factual)
        x_int[:, index] = value
        z_int = n_flow.transform(x_int)
        z_factual[:, index] = z_int[:, index]

        x_cf = n_flow.transform.inv(z_factual)
        self.flow.stop_intervening()

        return x_cf

    @torch.no_grad()
    def intervene_2(
        self, index: int, value: float, shape: Size = (), scaler=None
    ) -> Tensor:
        adj_int = self.adjacency.clone()
        adj_int[index, :] = 0.0
        self.flow.intervene(adj_int)

        n_flow = self.flow()
        if scaler is not None:  # x is not normalized
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        z1 = n_flow.base.rsample(shape)
        # Get distributional x1
        x1 = n_flow.transform.inv(z1)
        # Intervening on x
        x1[:, index] = value
        # Get interventional z
        z2 = n_flow.transform(x1)
        z1[:, index] = z2[:, index]

        x2 = n_flow.transform.inv(z1)

        self.flow.stop_intervening()
        return x2

    @torch.no_grad()
    def compute_counterfactual(
        self, x_factual, index: int, value: float, scaler=None, return_dict=False
    ) -> Tensor:
        output = {}
        n_flow = self.flow()
        if scaler is not None:  # x is not normalized
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        z_factual = n_flow.transform(x_factual)
        output["z_factual"] = z_factual.clone()
        x_tmp = x_factual.clone()
        x_tmp[:, index] = value
        z_cf = n_flow.transform(x_tmp)

        z_factual[:, index] = z_cf[:, index]

        x_cf = n_flow.transform.inv(z_factual)
        if return_dict:
            output["x_cf"] = x_cf
            output["z_cf"] = z_factual
            return output
        else:
            return x_cf

    @torch.no_grad()
    def compute_ate(self, index, a, b, num_samples=10000, scaler=None) -> Tensor:

        x_int = self.intervene(index, a, shape=(num_samples,), scaler=scaler)
        x_y = torch.cat((x_int[:, :index], x_int[:, index + 1 :]), dim=-1)

        mean_a = x_y.mean(0)

        x_int = self.intervene(index, b, shape=(num_samples,), scaler=scaler)
        x_y = torch.cat((x_int[:, :index], x_int[:, index + 1 :]), dim=-1)

        mean_b = x_y.mean(0)

        ate = mean_a - mean_b

        return ate

    @torch.no_grad()
    def intervene(
        self, index: int, value: float, shape: Size = (), scaler=None
    ) -> Tensor:

        n_flow = self.flow()
        if scaler is not None:  # x is not normalized
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        z1 = n_flow.base.rsample(shape)

        # Get distributional x1
        x1 = n_flow.transform.inv(z1)

        # Intervening on x
        x1[:, index] = value

        # Get interventional z
        z2 = n_flow.transform(x1)
        z2[:, index + 1 :] = z1[:, index + 1 :]

        x2 = n_flow.transform.inv(z2)

        return x2

    def stop_intervening(self, index: int) -> None:
        pass

    def intervening(self) -> bool:
        pass

    def log_prob(self, x: Tensor, scaler=None) -> Tensor:
        n_flow = self.flow()

        if scaler is not None:  # x is not normalized
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)
        log_prob = n_flow.log_prob(x)

        return log_prob

    def _inverse(self, x: Tensor) -> Tensor:
        n_flow = self.flow()
        u = n_flow.transform.inv(x)

        return u

    def compute_jacobian(self, x=None, u=None):
        jacobian_list = []

        n_flow = self.flow()

        with torch.enable_grad():
            if x is not None:  # Compute Jacobian at x
                fn = n_flow.transform
                jac = torch.autograd.functional.jacobian(fn, x.mean(0)).numpy()
            else:  # Compute Jacobian at u
                fn = n_flow.transform.inv
                jac = torch.autograd.functional.jacobian(fn, u.mean(0)).numpy()

        jacobian_list.append(jac)

        return jacobian_list

    def compute_jacobian_(self, x=None, u=None):
        jacobian_list = []

        if x is not None:  # Compute Jacobian at x
            input_ = x
            if self.flow.base_to_data:  # x = T(u)
                fn_name = "_inverse"
            else:  # u = T(x)
                fn_name = "_call"
        else:  # Compute Jacobian at u
            input_ = u
            if self.flow.base_to_data:  # x = T(u)
                fn_name = "_call"
            else:  # u = T(x)
                fn_name = "_inverse"

        if fn_name == "_inverse":
            transforms = self.flow.transforms[::-1]
        else:
            transforms = self.flow.transforms
        product = torch.eye(input_.shape[-1])
        for _, tran in enumerate(transforms):
            fn = getattr(tran(), fn_name)
            jac = torch.autograd.functional.jacobian(fn, input_.mean(dim=0))
            input_ = fn(input_)
            product = product @ jac.cpu()
            jacobian_list.append(jac.numpy())
        jacobian_list.append(product.numpy())

        return jacobian_list

    def log_abs_det_jacobian(self, u: Tensor, x: Tensor) -> Tensor:
        if self.derivatives is None:
            return self._log_abs_det_jacobian_autodiff(u, x)

        logdetjac = []
        for i, g in enumerate(self.derivatives):
            grad_i = g(*x[..., : i + 1].unbind(dim=-1))
            logdetjac.append(torch.log(grad_i.abs()))

        return torch.stack(logdetjac, dim=-1)
