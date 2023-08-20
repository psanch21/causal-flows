import os
import time

import matplotlib.pyplot as plt
import torch
import wandb
from tueplots import bundles

import causal_nf.utils.io as causal_io
from causal_nf.models.base_model import BaseLightning
from causal_nf.utils.graph import ancestor_matrix
from causal_nf.utils.optimizers import build_optimizer, build_scheduler

plt.rcParams.update(bundles.icml2022())
from causal_nf.utils.pairwise.mmd import maximum_mean_discrepancy
from causal_nf.modules.vaca import VACA

import numpy as np


class VACALightning(BaseLightning):
    def __init__(
        self,
        preparator,
        model: VACA,
        objective="elbo",
        beta=1.0,
        init_fn=None,
        plot=True,
    ):
        super(VACALightning, self).__init__(preparator, init_fn=init_fn)

        self.model = model
        self.plot = plot
        self.objective = objective
        self.beta = beta

        self.set_input_scaler()
        self.model.set_batch_generator(preparator.get_batch_generator())
        self.reset_parameters()

    def reset_parameters(self):
        super(VACALightning, self).reset_parameters()

    def set_input_scaler(self):
        self.input_scaler = self.preparator.get_scaler(fit=True)

    def forward(self, batch, **kwargs):
        tic = time.time()

        output = self.model(
            batch, beta=self.beta, objective=self.objective, scaler=self.input_scaler
        )

        output["time_forward"] = self.compute_time(tic, batch.num_graphs)
        return output

    def compute_time(self, tic, num_samples):
        delta_time = (time.time() - tic) * 1000
        return torch.tensor(delta_time / num_samples * 1000)

    @torch.no_grad()
    def predict(
        self,
        batch,
        observational=False,
        intervene=False,
        counterfactual=False,
        ate=False,
    ):
        output = {}
        num_graphs = batch.num_graphs
        x = batch.x.to(self.device).reshape(num_graphs, -1)
        with torch.enable_grad():
            output["log_prob_true"] = self.preparator.log_prob(x)

        tic = time.time()
        log_prob = self.model.log_prob(batch, scaler=self.input_scaler)
        output["time_log_prob"] = self.compute_time(tic, num_graphs)
        output["loss"] = -log_prob
        output["log_prob"] = log_prob

        if observational:
            tic = time.time()
            obs_dict = self.model.sample(num_graphs, scaler=self.input_scaler)
            output["time_sample_obs"] = self.compute_time(tic, num_graphs)
            x_obs = obs_dict["x_obs"]
            if self.plot:
                output["x"] = self.preparator.post_process(x)
            if self.plot:
                output["x_obs"] = self.preparator.post_process(x_obs)
            mmd_value = maximum_mean_discrepancy(x, x_obs, sigma=None)
            output[f"mmd_obs"] = mmd_value
            with torch.enable_grad():
                log_p_with_x_sample = self.preparator.log_prob(x_obs)
                log_p_with_x = self.preparator.log_prob(x)
            output["log_prob_p"] = log_p_with_x_sample
            log_q_with_x_sample = self.model.log_prob(batch, scaler=self.input_scaler)

            kl_distance = (
                log_p_with_x + log_q_with_x_sample - log_p_with_x_sample - log_prob
            )
            output["kl_distance"] = kl_distance

        if intervene:
            intervention_list = self.preparator.get_intervention_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                value = int_dict["value"]
                index = int_dict["index"]
                tic = time.time()
                x_int = self.model.intervene(
                    index=index,
                    value=value,
                    num_graphs=num_graphs,
                    scaler=self.input_scaler,
                )
                delta_times.append(self.compute_time(tic, num_graphs))

                if self.plot:
                    output[f"x_int_{index + 1}={name}"] = self.preparator.post_process(
                        x_int
                    )

                x_int_true = self.preparator.intervene(
                    index=index, value=value, shape=(num_graphs,)
                )

                if self.plot:
                    output[
                        f"x_int_{index + 1}={name}_true"
                    ] = self.preparator.post_process(x_int_true)

                mmd_value = maximum_mean_discrepancy(x_int, x_int_true, sigma=None)
                output[f"mmd_int_x{index + 1}={name}"] = mmd_value

            delta_time = torch.stack(delta_times).mean()
            output["time_intervene"] = delta_time
        if counterfactual:
            intervention_list = self.preparator.get_intervention_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                value = int_dict["value"]
                index = int_dict["index"]
                tic = time.time()
                x_cf = self.model.compute_counterfactual(
                    batch, index=index, value=value, scaler=self.input_scaler
                )
                delta_times.append(self.compute_time(tic, num_graphs))

                x_cf_true = self.preparator.compute_counterfactual(x, index, value)

                diff_cf = x_cf_true - x_cf

                rmse = torch.sqrt((diff_cf**2).sum(1))
                output[f"rmse_cf_x{index + 1}={name}"] = rmse
                mae = diff_cf.abs().sum(1)
                output[f"mse_cf_x{index + 1}={name}"] = mae

            delta_time = torch.stack(delta_times).mean()
            output["time_cf"] = delta_time

        if ate:
            intervention_list = self.preparator.get_ate_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                a = int_dict["a"]
                b = int_dict["b"]
                index = int_dict["index"]
                tic = time.time()
                ate = self.model.compute_ate(
                    index, a=a, b=b, num_graphs=10000, scaler=self.input_scaler
                )
                delta_times.append(self.compute_time(tic, 10000))

                ate_true = self.preparator.compute_ate(
                    index, a=a, b=b, num_samples=10000
                )

                diff_ate = ate_true - ate

                rmse = torch.sqrt((diff_ate**2).sum())
                output[f"rmse_ate_x{index + 1}={name}"] = rmse

            delta_time = torch.stack(delta_times).mean()
            output["time_ate"] = delta_time

        return output

    # process inside the training loop
    def training_step(self, train_batch, batch_idx):

        loss_dict = self(train_batch)

        loss_dict["loss"] = loss_dict["loss"].mean()
        log_dict = {}

        self.update_log_dict(log_dict=log_dict, my_dict=loss_dict, regex=r"^(?!x_).*$")

        return loss_dict

    def validation_step(self, batch, batch_idx):
        self.eval()

        if self.current_epoch % 10 == 1:
            observational = batch_idx == 0
            intervene = False
            ate = False
        else:
            observational = False
            intervene = False
            ate = False

        loss_dict = self.predict(
            batch,
            observational=observational,
            intervene=intervene,
            counterfactual=False,
            ate=ate,
        )

        log_dict = {}

        self.update_log_dict(
            log_dict=log_dict, my_dict=loss_dict, regex=r"^(?!.*x_).*$"
        )

        return log_dict

    def add_noise(self, x):
        # Calculate the standard deviation of each column
        std = torch.std(x, dim=0).mul(100).round() / 100.0

        # Find the columns that are constant (i.e., have a standard deviation of 0)
        constant_mask = std == 0
        # Generate a small amount of noise for each constant column
        noise = torch.randn(x.shape[0], sum(constant_mask)) * 0.01

        # Add the noise to the corresponding columns
        x[:, constant_mask] += noise
        return x

    def compute_metrics_stats(self, outputs):

        metric_stats = super(VACALightning, self).compute_metrics_stats(outputs)

        metric_stats = {
            key: value for key, value in metric_stats.items() if "x_" not in key
        }

        data = {}

        plot_intervene = False

        for output_i in outputs:
            for key, values in output_i.items():
                if "x" in key:
                    if key not in data:
                        data[key] = []
                    data[key].append(values)

                    if "x_int" in key:
                        plot_intervene = True

        n = 256

        split = self.preparator.current_split
        filename = os.path.join(self.logger.save_dir, f"split={split}_name=")
        if "x_obs" in data and split != "train":
            x_obs = data["x_obs"]
            x = data["x"]
            x_obs = torch.cat(x_obs, dim=0)[:n]
            x = torch.cat(x, dim=0)[:n]
            df = self.preparator.create_df([x, x_obs], ["real", "fake"])

            fig = self.preparator._plot_data(df=df, hue="mode")

            try:
                wandb.log({"x_obs": wandb.Image(fig)}, step=self.current_epoch)
            except:
                causal_io.print_warning("Could not log plot x_obs to wandb")

            fig.savefig(f"{filename}x_obs.png")
            plt.close("all")

        if plot_intervene and split != "train":
            for key in data:
                if "x_int" in key and "true" not in key:
                    x_int = data[key]
                    x_int_true = data[key + "_true"]
                    x_int = torch.cat(x_int, dim=0)[:n]
                    x_int_true = torch.cat(x_int_true, dim=0)[:n]

                    x_int = self.add_noise(x_int)
                    x_int_true = self.add_noise(x_int_true)

                    df = self.preparator.create_df(
                        [x_int_true, x_int], ["real", "fake"]
                    )
                    fig = self.preparator._plot_data(df=df, hue="mode")
                    try:
                        wandb.log({key: wandb.Image(fig)}, step=self.current_epoch)
                    except:
                        fig.savefig(f"{filename}{key}.png")

                    plt.close("all")
        return metric_stats

    def test_step(self, batch, batch_idx):

        self.eval()

        observational = batch_idx < 1
        intervene = batch_idx < 1
        counterfactual = batch_idx < 1
        ate = batch_idx < 1

        loss_dict = self.predict(
            batch,
            observational=observational,
            intervene=intervene,
            counterfactual=counterfactual,
            ate=ate,
        )

        log_dict = {}

        self.update_log_dict(log_dict=log_dict, my_dict=loss_dict)
        return log_dict

    def configure_optimizers(self):
        self.lr = self.optim_config.base_lr
        causal_io.print_debug(f"Setting lr: {self.lr}")

        params = self.model.parameters()
        opt = build_optimizer(optim_config=self.optim_config, params=params)

        output = {}

        if isinstance(self.optim_config.scheduler, str):
            sched = build_scheduler(optim_config=self.optim_config, optimizer=opt)
            output["optimizer"] = opt
            output["lr_scheduler"] = sched
            output["monitor"] = "val_loss"
        else:
            output["optimizer"] = opt
        return output

    def plot(self):
        raise NotImplementedError

    def _plot_jacobian(self, J, title="Jacobian Matrix", variable="x"):
        if isinstance(J, torch.Tensor):
            J = J.detach().numpy()

        J_abs = np.absolute(J)
        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Plot the matrix using the axis object's `matshow` function
        height, width = J.shape
        fig_aspect_ratio = fig.get_figheight() / fig.get_figwidth()
        data_aspect_ratio = (height / width) * fig_aspect_ratio
        # Plot the matrix using the axis object's `matshow` function
        cax = ax.matshow(
            J_abs, aspect=data_aspect_ratio, cmap="viridis"
        )  # You can change the colormap to your preference

        # Add a colorbar to the plot for easy interpretation
        fig.colorbar(cax)

        # Set the title for the axis object
        ax.set_title(f"{title} {variable}")

        # Label the x and y axes
        ax.set_xticks(range(J.shape[1]))
        ax.set_yticks(range(J.shape[0]))

        xticks = [
            "$\\frac{{ \\partial f_m }}{{ \\partial {}_{} }}$".format(variable, i)
            for i in range(1, J.shape[1] + 1)
        ]
        ax.set_xticklabels(xticks)
        yticks = [
            "$\\frac{{ \\partial f_{} }}{{ \\partial {}_n }}$".format(i, variable)
            for i in range(1, J.shape[1] + 1)
        ]
        ax.set_yticklabels(yticks)

        # Display the values of the Jacobian matrix with 2 decimal points
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                value = J[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")

        return fig
