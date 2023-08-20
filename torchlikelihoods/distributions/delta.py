import torch
import torch.distributions as td


class Delta(td.Distribution):
    def __init__(self, center=None, lambda_=1.0, validate_args=None):
        if center is None:
            raise ValueError("`center` must be specified.")
        self.center = center
        self.lambda_ = lambda_
        self._param = self.center
        batch_shape = self._param.size()
        super(Delta, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.center

    def sample(self, sample_shape=torch.Size()):
        return self.center

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        return -(1 / self.lambda_) * (value - self.center) ** 2
