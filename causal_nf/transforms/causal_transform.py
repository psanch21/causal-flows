import abc
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Transform, constraints


class CausalTransform(Transform, abc.ABC):
    @abc.abstractmethod
    def intervene(self, index: int, value: float) -> None:
        pass

    @abc.abstractmethod
    def stop_intervening(self, index: int) -> None:
        pass

    @abc.abstractmethod
    def intervening(self) -> bool:
        pass


class CausalEquations(CausalTransform):

    domain = constraints.unit_interval
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, functions, inverses, derivatives=None):
        super(CausalEquations, self).__init__(cache_size=0)
        self.functions = functions
        self.inverses = inverses
        self.derivatives = derivatives

        self._interventions = dict()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CausalEquations)

    def _call(self, u: Tensor) -> Tensor:
        assert u.shape[1] == len(self.functions)

        x = []
        for i, f in enumerate(self.functions):
            if i in self._interventions:
                x_i = torch.ones_like(u[..., i]) * self._interventions[i]
            else:
                x_i = f(*x[:i], u[..., i])
            x.append(x_i)
        x = torch.stack(x, dim=1)

        return x

    def _inverse(self, x: Tensor) -> Tensor:
        assert x.shape[1] == len(self.inverses)

        u = []
        for i, g in enumerate(self.inverses):
            u_i = g(*x[..., : i + 1].unbind(dim=-1))
            u.append(u_i)
        u = torch.stack(u, dim=1)

        return u

    def log_abs_det_jacobian(self, u: Tensor, x: Tensor) -> Tensor:
        if self.derivatives is None:
            return self._log_abs_det_jacobian_autodiff(u, x)

        logdetjac = []
        for i, g in enumerate(self.derivatives):
            grad_i = g(*x[..., : i + 1].unbind(dim=-1))
            logdetjac.append(torch.log(grad_i.abs()))

        return -torch.stack(logdetjac, dim=-1)

    def _log_abs_det_jacobian_autodiff(
        self, u: Tensor, x: Tensor
    ) -> Tensor:
        logdetjac = []
        old_requires_grad = x.requires_grad
        x.requires_grad_(True)
        for i, g in enumerate(self.inverses):  # u = T(x)
            u_i = g(*x[..., : i + 1].unbind(dim=-1))
            grad_i = torch.autograd.grad(u_i.sum(), x)[0][..., i]
            logdetjac.append(torch.log(grad_i.abs()))
        x.requires_grad_(old_requires_grad)
        return -torch.stack(logdetjac, dim=-1)

    def intervene(self, index, value) -> None:
        self._interventions[index] = value

    def stop_intervening(self, index: int) -> None:
        self._interventions.pop(index)

    @property
    def intervening(self) -> bool:
        return len(self._interventions) > 0
