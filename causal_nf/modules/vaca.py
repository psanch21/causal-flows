import torch
import torch.nn as nn
from torch import Tensor, Size

from zuko.transforms import ComposedTransform
from torch_geometric.data import Batch, Data


def elbo(x, qz_x, pz, px_z, num_graphs, beta=1.0):
    kl_z_ = torch.distributions.kl.kl_divergence(qz_x, pz)
    kl_z_all = torch.reshape(kl_z_, shape=(num_graphs, -1))
    log_px_z_all = torch.reshape(px_z.log_prob(x), shape=(num_graphs, -1))

    assert kl_z_all.ndim == 2
    assert log_px_z_all.ndim == 2

    kl_z = kl_z_all.sum(1)
    log_px_z = log_px_z_all.sum(1)

    elbo = log_px_z - beta * kl_z

    data = {"log_px_z": log_px_z, "kl_z": kl_z, "elbo": elbo, "objective": elbo}

    return data


class VACA(nn.Module):
    def __init__(
        self, encoder_gnn, decoder_gnn, prior_distr, posterior_distr, likelihood_distr
    ):
        super(VACA, self).__init__()

        self.encoder_gnn = encoder_gnn
        self.decoder_gnn = decoder_gnn

        self.prior_distr = prior_distr

        self.posterior_distr = posterior_distr

        self.z_dim = self.posterior_distr.domain_size()
        self.likelihood_distr = likelihood_distr

        self.batch_generator = None

    def get_index_z_intervention(self, index):
        index_list = list(range(index * self.z_dim, (index + 1) * self.z_dim))
        return index_list

    def set_batch_generator(self, batch_generator):
        self.batch_generator = batch_generator

    def forward(self, batch, beta=1.0, objective="elbo", scaler=None) -> Tensor:

        batch = batch.clone()
        num_graphs = batch.num_graphs
        num_nodes = batch.x.shape[0]
        if scaler is not None:
            batch.x = scaler.transform(batch.x.reshape(num_graphs, -1)).reshape(
                num_nodes, -1
            )

        x = batch.x
        logits = self.encoder_gnn(batch)
        qz_x = self.posterior_distr(logits)
        z = qz_x.rsample()
        batch.x = z
        logits = self.decoder_gnn(batch)
        px_z = self.likelihood_distr(logits)
        if objective == "elbo":
            output = elbo(
                x, qz_x, self.prior_distr, px_z, num_graphs=batch.num_graphs, beta=beta
            )
        else:
            raise NotImplementedError(f"objective {objective} not implemented")

        output["loss"] = -output["objective"]

        return output

    @torch.no_grad()
    def sample(self, num_graphs, scaler=None):
        assert self.batch_generator is not None
        batch = self.batch_generator(num_graphs)
        num_samples = batch.x.shape[0]
        output = {}
        z = self.prior_distr.sample((num_samples,))
        batch.x = z
        logits = self.decoder_gnn(batch)
        px_z = self.likelihood_distr(logits)

        x = px_z.sample().reshape((num_graphs, -1))

        if scaler is not None:
            x = scaler.inverse_transform(x)

        output["z_obs"] = z.reshape((num_graphs, -1))
        output["x_obs"] = x
        return output

    @torch.no_grad()
    def reconstruct(self, batch, scaler=None):
        batch = batch.clone()
        num_graphs = batch.num_graphs
        x = batch.x.reshape((num_graphs, -1))
        num_nodes = batch.x.shape[0]
        if scaler is not None:
            batch.x = scaler.transform(batch.x.reshape(num_graphs, -1)).reshape(
                num_nodes, -1
            )
        x_norm = batch.x
        logits = self.encoder_gnn(batch)
        z_mean, qz_x = self.posterior_distr(logits, return_mean=True)

        batch.x = z_mean

        logits = self.decoder_gnn(batch)
        x_mean, px_z = self.likelihood_distr(logits, return_mean=True)
        output = {}
        output["z_rec"] = z_mean.reshape((num_graphs, -1))
        output["x_rec"] = x_mean.reshape((num_graphs, -1))
        output["log_prob_x_rec"] = (
            px_z.log_prob(x_mean).reshape((num_graphs, -1)).sum(1)
        )
        output["log_prob_x_rec"] = (
            px_z.log_prob(x_norm).reshape((num_graphs, -1)).sum(1)
        )

        return output

    @torch.no_grad()
    def log_prob(self, batch, scaler=None):
        batch = batch.clone()
        num_graphs = batch.num_graphs
        x = batch.x.reshape((num_graphs, -1))
        num_nodes = batch.x.shape[0]
        if scaler is not None:
            batch.x = scaler.transform(batch.x.reshape(num_graphs, -1)).reshape(
                num_nodes, -1
            )
        x_norm = batch.x
        logits = self.encoder_gnn(batch)
        z_mean, qz_x = self.posterior_distr(logits, return_mean=True)

        batch.x = z_mean

        logits = self.decoder_gnn(batch)
        px_z = self.likelihood_distr(logits, return_mean=False)
        log_prob = px_z.log_prob(x_norm).reshape((num_graphs, -1)).sum(1)

        output = elbo(
            x_norm, qz_x, self.prior_distr, px_z, num_graphs=batch.num_graphs, beta=1.0
        )

        return output["elbo"]

    @torch.no_grad()
    def compute_counterfactual(
        self, batch, index: int, value: float, scaler=None, return_dict=False
    ) -> Tensor:
        output = {}
        batch = batch.clone()
        batch_i = self.batch_generator.intervene(batch, index, value)

        num_graphs = batch.num_graphs
        num_nodes_total = batch.x.shape[0]
        num_nodes = num_nodes_total // num_graphs
        if scaler is not None:
            batch.x = scaler.transform(batch.x.reshape(num_graphs, -1)).reshape(
                num_nodes_total, -1
            )
            batch_i.x = scaler.transform(batch_i.x.reshape(num_graphs, -1)).reshape(
                num_nodes_total, -1
            )
        logits = self.encoder_gnn(batch)
        z_factual, qz_x = self.posterior_distr(logits, return_mean=True)
        z_factual = z_factual.reshape((num_graphs, -1))
        output["z_factual"] = z_factual
        logits = self.encoder_gnn(batch_i)
        z_cf, qz_x = self.posterior_distr(logits, return_mean=True)
        z_cf = z_cf.reshape((num_graphs, -1))
        output["z_cf"] = z_cf

        index_list = self.get_index_z_intervention(index)
        z_factual[:, index_list] = z_cf[:, index_list]
        output["z_dec"] = z_factual
        z_factual = z_factual.reshape((num_nodes_total, -1))

        assert (
            z_factual[index::num_nodes] == z_factual[index::num_nodes, :]
        ).all(), f"z_factual[index,: ] != z_factual[index::num_nodes, :]"
        batch_i.x = z_factual
        logits = self.decoder_gnn(batch_i)
        assert (
            logits[index::num_nodes, :] == logits[index::num_nodes, :]
        ).all(), f"logits[index,: ] != logits[index::num_nodes, :]"
        x_cf, px_z = self.likelihood_distr(logits, return_mean=True)
        x_cf = x_cf.reshape(num_graphs, -1)
        if scaler is not None:
            x_cf = scaler.inverse_transform(x_cf)
        output["x_cf"] = x_cf

        if return_dict:
            return output
        else:
            return x_cf

    @torch.no_grad()
    def compute_ate(self, index, a, b, num_graphs=10000, scaler=None) -> Tensor:

        x_int = self.intervene(index, a, num_graphs=num_graphs, scaler=scaler)
        x_y = torch.cat((x_int[:, :index], x_int[:, index + 1 :]), dim=-1)

        mean_a = x_y.mean(0)

        x_int = self.intervene(index, b, num_graphs=num_graphs, scaler=scaler)
        x_y = torch.cat((x_int[:, :index], x_int[:, index + 1 :]), dim=-1)

        mean_b = x_y.mean(0)

        ate = mean_a - mean_b

        return ate

    @torch.no_grad()
    def intervene(self, index: int, value: float, num_graphs, scaler=None) -> Tensor:
        assert self.batch_generator is not None
        batch = self.batch_generator(num_graphs)
        num_nodes = batch.x.shape[0]
        batch_i = self.batch_generator.intervene(batch, index, value)
        logits = self.encoder_gnn(batch_i)
        z_i, qz_x = self.posterior_distr(logits, return_mean=True)
        z_i = z_i.reshape((num_graphs, -1))

        z = self.prior_distr.sample((num_nodes,))
        z = z.reshape((num_graphs, -1))
        index_list = self.get_index_z_intervention(index)
        z[:, index_list] = z_i[:, index_list]

        batch_i.x = z.reshape((num_nodes, -1))
        logits = self.decoder_gnn(batch_i)
        x_int, px_z = self.likelihood_distr(logits, return_mean=True)
        x_int = x_int.reshape(num_graphs, -1)

        if scaler is not None:
            x_int = scaler.inverse_transform(x_int)

        return x_int

    def stop_intervening(self, index: int) -> None:
        pass

    def intervening(self) -> bool:
        pass
