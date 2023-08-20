from torch.distributions import Distribution, Normal
import torch


class Heterogeneous(Distribution):
    def __init__(self, distr_list, **kwargs):
        if "batch_shape" not in kwargs:
            kwargs["batch_shape"] = (len(distr_list),)
        super(Heterogeneous, self).__init__(**kwargs)

        self.distr_list = distr_list
        self.num_dim = len(self.distr_list)

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        mean = []
        for distr in self.distr_list:
            mean.append(distr.mean)

        return torch.cat(mean, dim=-1)

    @property
    def mode(self):
        """
        Returns the mode of the distribution.
        """
        mode = []
        for distr in self.distr_list:
            mode.append(distr.mode)

        return torch.cat(mode, dim=-1)

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        std = []
        for distr in self.distr_list:
            std.append(distr.variance)

        return torch.cat(std, dim=-1)

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        samples = []
        for distr in self.distr_list:
            samples.append(distr.sample(sample_shape))

        return torch.cat(samples, dim=-1)

    def expand(self, batch_shape, _instance=None):
        new_ = []
        idx_init = 0
        idx_end = 0
        for distr in self.distr_list:
            dim_i = distr.batch_shape[0]
            if len(batch_shape) == 1:
                batch_shape_i = torch.Size([dim_i])
            elif len(batch_shape) == 2:
                batch_shape_i = torch.Size([batch_shape[0], dim_i])

            new_.append(distr.expand(batch_shape_i))

        return Heterogeneous(new_)

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        log_prob = []
        idx_init = 0
        idx_end = 0
        for distr in self.distr_list:
            dim_i = distr.batch_shape[0]
            idx_end += dim_i
            value_i = value[:, idx_init:idx_end]
            log_prob.append(distr.log_prob(value_i))
            idx_init = idx_end

        return torch.cat(log_prob, dim=-1)
