import glob

import wandb
import os
import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
import causal_nf.utils.wandb_local as wandb_local
from causal_nf.config import cfg
import causal_nf.utils.io as causal_nf_io


os.environ["WANDB_NOTEBOOK_NAME"] = "name_of_the_notebook"

args_list, args = causal_nf_config.parse_args()

load_model = isinstance(args.load_model, str)
if load_model:
    causal_nf_io.print_info(f"Loading model: {args.load_model}")

config = causal_nf_config.build_config(
    config_file=args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

causal_nf_config.assert_cfg_and_config(cfg, config)


if cfg.device in ["cpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
causal_nf_train.set_reproducibility(cfg)

if cfg.dataset.name in ["german"]:
    from causal_nf.preparators.german_preparator import GermanPreparator

    preparator = GermanPreparator.loader(cfg.dataset)
elif cfg.dataset.name in ["ihdp"]:
    from causal_nf.preparators.ihdp_preparator import IHDPPreparator

    preparator = IHDPPreparator.loader(cfg.dataset)
else:
    from causal_nf.preparators.scm import SCMPreparator

    preparator = SCMPreparator.loader(cfg.dataset)
preparator.prepare_data()

loaders = preparator.get_dataloaders(
    batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
)

for i, loader in enumerate(loaders):
    causal_nf_io.print_info(f"[{i}] num_batchees: {len(loader)}")

model = causal_nf_train.load_model(cfg=cfg, preparator=preparator)

param_count = model.param_count()
config["param_count"] = param_count

if not load_model:
    assert isinstance(args.project, str)
    run = wandb.init(
        mode=args.wandb_mode,
        group=args.wandb_group,
        project=args.project,
        config=config,
    )

    import uuid

    if args.wandb_mode != "disabled":
        run_uuid = run.id
    else:
        run_uuid = str(uuid.uuid1()).replace("-", "")
else:
    run_uuid = os.path.basename(args.load_model)

# # # Here you can add many features in your Trainer: such as num_epochs,  gpus used, clusters used etc.

dirpath = os.path.join(cfg.root_dir, run_uuid)

if load_model:
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_dir = os.path.join(cfg.root_dir, run_uuid, "evaluate", now)
else:
    logger_dir = os.path.join(cfg.root_dir, run_uuid)

trainer, logger = causal_nf_train.load_trainer(
    cfg=cfg,
    dirpath=dirpath,
    logger_dir=logger_dir,
    include_logger=True,
    model_checkpoint=cfg.train.model_checkpoint,
    cfg_early=cfg.early_stopping,
    preparator=preparator,
)

causal_nf_io.print_info(f"Experiment folder: {logger.save_dir}\n\n")

wandb_local.log_config(dict(config), root=logger.save_dir)

if not load_model:
    wandb_local.copy_config(
        config_default=causal_nf_config.DEFAULT_CONFIG_FILE,
        config_experiment=args.config_file,
        root=logger.save_dir,
    )
    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])

if isinstance(preparator.single_split, str):
    loaders = [loaders[0]]

model.save_dir = dirpath

if load_model:
    ckpt_name_list = glob.glob(os.path.join(args.load_model, f"*ckpt"))
    for ckpt_file in ckpt_name_list:
        model = causal_nf_train.load_model(
            cfg=cfg, preparator=preparator, ckpt_file=ckpt_file
        )
        model.eval()
        model.save_dir = dirpath
        ckpt_name = preparator.get_ckpt_name(ckpt_file)
        for i, loader_i in enumerate(loaders):
            s_name = preparator.split_names[i]
            causal_nf_io.print_info(f"Testing {s_name} split")
            preparator.set_current_split(i)
            model.ckpt_name = ckpt_name
            _ = trainer.test(model=model, dataloaders=loader_i)
            metrics_stats = model.metrics_stats
            metrics_stats["current_epoch"] = trainer.current_epoch
            wandb_local.log_v2(
                {s_name: metrics_stats, "epoch": ckpt_name},
                root=trainer.logger.save_dir,
            )


else:

    ckpt_name_list = ["last"]
    if cfg.early_stopping.activate:
        ckpt_name_list.append("best")
    for ckpt_name in ckpt_name_list:
        for i, loader_i in enumerate(loaders):
            s_name = preparator.split_names[i]
            causal_nf_io.print_info(f"Testing {s_name} split")
            preparator.set_current_split(i)
            model.ckpt_name = ckpt_name
            _ = trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
            metrics_stats = model.metrics_stats
            metrics_stats["current_epoch"] = trainer.current_epoch

            wandb_local.log_v2(
                {s_name: metrics_stats, "epoch": ckpt_name},
                root=trainer.logger.save_dir,
            )

    run.finish()
    if args.delete_ckpt:
        for f in glob.iglob(os.path.join(logger.save_dir, "*.ckpt")):
            causal_nf_io.print_warning(f"Deleting {f}")
            os.remove(f)

print(f"Experiment folder: {logger.save_dir}")
