import torch.optim as optim

from causal_nf.utils.io import dict_to_cn


def build_optimizer(optim_config, params):
    if isinstance(optim_config, dict):
        optim_config = dict_to_cn(optim_config)

    params = filter(lambda p: p.requires_grad, params)
    # Try to load customized optimizer

    if optim_config.optimizer == "adam":
        betas = (optim_config.beta_1, optim_config.beta_2)
        optimizer = optim.Adam(
            params,
            lr=optim_config.base_lr,
            betas=betas,
            weight_decay=optim_config.weight_decay,
        )
    elif optim_config.optimizer == "radam":
        optimizer = optim.RAdam(
            params, lr=optim_config.base_lr, weight_decay=optim_config.weight_decay
        )
    elif optim_config.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr=optim_config.base_lr,
            momentum=optim_config.momentum,
            weight_decay=optim_config.weight_decay,
        )
    else:
        raise ValueError("Optimizer {} not supported".format(optim_config.optimizer))

    return optimizer


def build_scheduler(optim_config, optimizer):
    # Try to load customized scheduler

    if optim_config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=optim_config.step_size, gamma=optim_config.gamma
        )
    elif optim_config.scheduler == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=optim_config.gamma
        )
    elif optim_config.scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optim_config.max_epoch
        )
    elif optim_config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=optim_config.mode,
            factor=optim_config.factor,
            patience=optim_config.patience,
            cooldown=optim_config.cooldown,
        )
    else:
        raise ValueError("Scheduler {} not supported".format(optim_config.scheduler))
    return scheduler
