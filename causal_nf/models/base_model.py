from datetime import datetime

import pytorch_lightning as pl
import torch
import wandb

import causal_nf.utils.wandb_local as wandb_local
import torch.optim.lr_scheduler as t_lr

import re


class BaseLightning(pl.LightningModule):
    def __init__(self, preparator, init_fn=None):
        super(BaseLightning, self).__init__()
        self.preparator = preparator
        self.init_fn = init_fn
        self.model = None

        self.optim_config = None
        self.log_std = False

        self.metrics_stats = None
        self.ckpt_name = "unknown"
        self.save_dir = None

    def get_now(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return now

    def reset_parameters(self):
        if self.init_fn is not None:
            self.model.apply(self.init_fn)
        return

    def param_count(self):
        return sum([p.numel() for p in self.parameters()])

    def set_optim_config(self, config):
        self.optim_config = config

    def on_fit_start(self) -> None:
        self.input_scaler.to(self.device)
        self.preparator.on_start(self.device)

    def on_test_start(self) -> None:
        self.input_scaler.to(self.device)
        self.preparator.on_start(self.device)

    def update_log_dict(self, log_dict, my_dict, key_id="", regex=None):
        for key, value in my_dict.items():
            if isinstance(value, list):
                value_tensor = torch.cat(value)
            elif isinstance(value, torch.Tensor):
                value_tensor = value
            else:
                value_tensor = torch.tensor(value)

            if value_tensor.ndim == 0:
                value_tensor = value_tensor.unsqueeze(0)

            my_key = f"{key}{key_id}"

            log_dict[my_key] = value_tensor.detach()
            if isinstance(regex, str) and re.search(regex, key):
                self.log(my_key, log_dict[my_key].float().mean().item(), prog_bar=True)

    def set_input_scaler(self):
        raise NotImplementedError

    def compute_metrics_stats(self, outputs):

        metrics = {}
        for output in outputs:
            for key, values in output.items():
                if values.ndim == 0:
                    values = values.unsqueeze(0)
                if key not in metrics:
                    metrics[key] = None
                if metrics[key] is None:
                    metrics[key] = values
                else:

                    metrics[key] = torch.cat([metrics[key], values], dim=0)
        metrics_stats = {}
        for metric, values in metrics.items():
            if self.__is_metric(metric):
                if values.dtype in [torch.bool]:
                    values = values.float()
                if values.dtype != torch.float:
                    continue
                metrics_stats[metric] = values.mean().item()
                if self.log_std:
                    metrics_stats[f"{metric}_std"] = values.std().item()

        metrics_2 = self.preparator.compute_metrics(**metrics)
        metrics_stats.update(metrics_2)
        return metrics_stats

    def __is_metric(self, metric):
        cond1 = metric not in ["logits", "label", "target"]
        cond2 = "logits" not in metric
        return cond1 and cond2

    def training_epoch_end(self, outputs) -> None:

        metrics_stats = self.compute_metrics_stats(outputs)
        opt = self.optimizers()
        if isinstance(opt, list):
            for i, o in enumerate(opt):
                metrics_stats[f"lr_{i}"] = o.optimizer.param_groups[0]["lr"]
        else:
            metrics_stats[f"lr"] = opt.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        output = {"train": metrics_stats, "epoch": self.current_epoch}

        if isinstance(sch, list):
            for sch_i in sch:
                if not isinstance(sch_i, t_lr.ReduceLROnPlateau):
                    sch_i.step()
        elif sch is not None and not isinstance(sch, t_lr.ReduceLROnPlateau):
            sch.step()

        wandb.log(output, step=self.current_epoch)
        wandb_local.log_v2(output, root=self.logger.save_dir)

    def validation_epoch_end(self, outputs):
        metrics_stats = self.compute_metrics_stats(outputs)

        output = {"val": metrics_stats, "epoch": self.current_epoch}
        self.metrics_stats = output

        wandb.log(output, step=self.current_epoch)
        wandb_local.log_v2(output, root=self.logger.save_dir)
        sch = self.lr_schedulers()

        monitor = metrics_stats["loss"]
        if isinstance(sch, list):
            for sch_i in sch:
                if isinstance(sch_i, t_lr.ReduceLROnPlateau):
                    sch_i.step(monitor)
        elif sch is not None and isinstance(sch, t_lr.ReduceLROnPlateau):
            sch.step(monitor)

        for name, value in metrics_stats.items():
            self.log(f"val_{name}", value, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        log_dict = {}
        return log_dict

    def test_epoch_end(self, outputs):
        metrics_stats = self.compute_metrics_stats(outputs)

        self.metrics_stats = metrics_stats
        return
