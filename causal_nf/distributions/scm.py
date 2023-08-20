from torch.distributions import TransformedDistribution
from causal_nf.transforms import CausalTransform
import torch


class SCM(TransformedDistribution):
    def __init__(self, base_distribution, transform, validate_args=None):
        super(SCM, self).__init__(base_distribution, transform, validate_args)
        assert isinstance(transform, CausalTransform), "A CausalTransform is expected."

    @property
    def transform(self) -> CausalTransform:
        return self.transforms[0]

    def intervene(self, index, value):
        self.transform.intervene(index, value)

    def stop_intervening(self, index):
        self.transform.stop_intervening(index)

    def sample_counterfactual(self, factual):
        assert self.transform.intervening
        u_f = self.transform.inv(factual)
        x_cf = self.transform(u_f)
        return x_cf

    def compute_ate(self, index, a, b, num_samples=10000):
        self.intervene(index, a)

        x_int = self.sample((num_samples,))
        x_y = torch.cat((x_int[:, :index], x_int[:, index + 1 :]), dim=-1)

        mean_a = x_y.mean(0)

        self.stop_intervening(index)

        self.intervene(index, b)

        x_int = self.sample((num_samples,))
        x_y = torch.cat((x_int[:, :index], x_int[:, index + 1 :]), dim=-1)

        mean_b = x_y.mean(0)

        self.stop_intervening(index)

        ate = mean_a - mean_b

        return ate
