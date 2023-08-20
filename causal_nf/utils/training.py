import copy

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import causal_nf.utils.io as causal_io
import zuko.flows as zflows
from causal_nf.models.causal_nf import CausalNFightning
from causal_nf.models.vaca import VACALightning
from causal_nf.modules import module_dict
from causal_nf.modules.causal_nf import CausalNormalizingFlow
from causal_nf.modules.vaca import VACA
from causal_nf.utils.init import get_init_fn


def set_reproducibility(cfg):
    # Setting the seed
    pl.seed_everything(cfg.seed, workers=True)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    if cfg.device != "cpu":
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False


def load_preparator(cfg, prepare_data=True):
    return


def _load(ckpt_file, Model, **model_args):
    if isinstance(ckpt_file, str):
        causal_io.print_info(f"Loading {Model} from {ckpt_file}")
        model = Model.load_from_checkpoint(checkpoint_path=ckpt_file, **model_args)
    else:
        model = Model(**model_args)

    return model


def load_model(cfg, preparator, ckpt_file=None):
    if cfg.model.name in ["causal_nf", "carefl"]:
        model = load_normalizing_flow(cfg, preparator, ckpt_file=ckpt_file)
    elif cfg.model.name == "vaca":
        model = load_vaca(cfg, preparator, ckpt_file=ckpt_file)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented")
    model.set_optim_config(cfg.optim)

    return model


from torch.distributions import Normal
from torchlikelihoods.likelihoods import NormalLikelihood, DeltaLikelihood


def load_vaca(cfg, preparator, ckpt_file=None):
    init_fn = get_init_fn(cfg_model=cfg.model)
    GNN = module_dict[cfg.model.layer_name]
    latent_dim = cfg.model.latent_dim

    posterior_distr = NormalLikelihood(latent_dim)

    cfg_copy = copy.deepcopy(cfg)
    kwargs = GNN.kwargs(
        cfg_copy,
        preparator,
        input_size=preparator.x_dim(),
        output_size=posterior_distr.params_size(),
    )
    encoder = GNN(**kwargs)
    print(f"-----------------------")
    print(f"-      ENCODER        -")
    print(f"-----------------------")
    print(encoder)
    print(posterior_distr)

    pz = Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
    print(f"-----------------------")
    print(f"-      PRIOR          -")
    print(f"-----------------------")
    print(pz)
    if cfg.model.distr_x == "normal":
        likelihood_distr = NormalLikelihood(preparator.x_dim())
    elif cfg.model.distr_x == "delta":
        Delta = DeltaLikelihood.create(lambda_=cfg.model.lambda_)
        likelihood_distr = Delta(preparator.x_dim())
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.gnn = cfg_copy.gnn2

    if cfg_copy.gnn.num_layers == 0:
        cfg_copy.gnn.num_layers = preparator.diameter()
    kwargs = GNN.kwargs(
        cfg_copy,
        preparator,
        input_size=cfg.model.latent_dim,
        output_size=likelihood_distr.params_size(),
    )
    decoder = GNN(**kwargs)
    print(f"-----------------------")
    print(f"-      DECODER        -")
    print(f"-----------------------")
    print(decoder)
    print(likelihood_distr)

    vaca = VACA(
        encoder_gnn=encoder,
        decoder_gnn=decoder,
        prior_distr=pz,
        posterior_distr=posterior_distr,
        likelihood_distr=likelihood_distr,
    )
    model = _load(
        ckpt_file=ckpt_file,
        Model=VACALightning,
        preparator=preparator,
        model=vaca,
        objective=cfg.model.objective,
        beta=cfg.model.beta,
        init_fn=init_fn,
        plot=cfg.model.plot,
    )
    return model


def load_normalizing_flow(cfg, preparator, ckpt_file=None):
    init_fn = get_init_fn(cfg_model=cfg.model)
    flow_name = cfg.model.layer_name
    hidden_features = cfg.model.dim_inner
    dim_x = preparator.x_dim()

    if cfg.model.num_layers == -1:
        num_layers = preparator.longest_path_length()
    elif cfg.model.num_layers == 0:
        num_layers = preparator.diameter()
    else:

        num_layers = cfg.model.num_layers
    base_to_data = cfg.model.base_to_data
    activation = cfg.model.act
    if cfg.model.adjacency:
        adjacency = preparator.adjacency()
    else:
        adjacency = None

    base_distr = cfg.model.base_distr

    learn_base = cfg.model.learn_base

    activation = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "lrelu": torch.nn.LeakyReLU,
        "sigmoid": torch.nn.Sigmoid,
    }[activation]

    model_name = cfg.model.name

    if model_name == "causal_nf":

        if flow_name == "maf":
            flow = zflows.MAF(
                dim_x,
                0,
                transforms=num_layers,
                hidden_features=hidden_features,
                adjacency=adjacency,
                base_to_data=base_to_data,
                base_distr=base_distr,
                learn_base=learn_base,
                activation=activation,
            )
        elif flow_name == "unaf":
            flow = zflows.UNAF(
                dim_x, 0, transforms=num_layers, hidden_features=hidden_features
            )
        elif flow_name == "nsf":
            flow = zflows.NSF(
                dim_x,
                0,
                transforms=num_layers,
                hidden_features=hidden_features,
                adjacency=adjacency,
                base_to_data=base_to_data,
                base_distr=base_distr,
                learn_base=learn_base,
                activation=activation,
            )
        elif flow_name == "naf":
            flow = zflows.NAF(
                features=dim_x,
                context=0,
                transforms=num_layers,
                hidden_features=hidden_features,
                randperm=False,
                activation=activation,
            )
        else:
            raise NotImplementedError(f"Flow {flow_name} not implemented")

        print(flow)
        module = CausalNormalizingFlow(flow=flow)

    model = _load(
        ckpt_file=ckpt_file,
        Model=CausalNFightning,
        preparator=preparator,
        model=module,
        init_fn=init_fn,
        plot=cfg.model.plot,
        regularize=cfg.train.regularize,
        kl=cfg.train.kl,
    )

    model.set_optim_config(cfg.optim)
    return model


def load_trainer(
    cfg,
    dirpath,
    logger_dir=None,
    include_logger=True,
    model_checkpoint=True,
    cfg_early=None,
    preparator=None,
):
    if logger_dir is None:
        logger_dir = dirpath
    devices = None
    if torch.cuda.is_available() and cfg.device in ["gpu", "auto"]:
        devices = torch.cuda.device_count()

        causal_io.print_info(f"Using {devices} GPUs")
        for i in range(devices):
            causal_io.print_info(f"[{i}] {torch.cuda.get_device_name(i)}")

        if devices == 0:
            devices = 1
    callbacks = []
    if preparator is not None:
        monitor, mode = preparator.monitor()
    else:
        monitor = None
        mode = "min"

    if model_checkpoint:
        model_checkpoint = ModelCheckpoint(
            dirpath=dirpath,
            every_n_epochs=None,
            save_last=True,
            save_top_k=1,
            monitor=monitor,
            mode=mode,
            save_weights_only=True,
        )
        callbacks.append(model_checkpoint)
    if cfg_early is not None and cfg_early.activate:
        early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=cfg_early.min_delta,
            patience=cfg_early.patience,
            verbose=cfg_early.verbose,
            mode=mode,
            check_on_train_epoch_end=False,
        )
        callbacks.append(early_stop_callback)

    from pytorch_lightning.loggers import CSVLogger

    if include_logger:
        logger = CSVLogger(save_dir=logger_dir, name="logs")
    else:
        logger = None

    trainer = pl.Trainer(
        default_root_dir=cfg.root_dir,
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
        devices=devices,
        auto_select_gpus=True,
        accelerator=cfg.device,
        auto_scale_batch_size=cfg.train.auto_scale_batch_size,
        max_epochs=cfg.train.max_epochs,
        profiler=cfg.train.profiler,
        enable_progress_bar=cfg.train.enable_progress_bar,
        max_time=cfg.train.max_time,
        # auto_lr_find=cfg.train.auto_lr_find,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        fast_dev_run=False,
        inference_mode=cfg.train.inference_mode,
    )
    if include_logger:
        return trainer, logger
    else:
        return trainer
