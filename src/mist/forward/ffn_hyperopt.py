"""ffn_hyperopt.py

Hyperopt parameters

"""
import os
import copy
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler

# from ray.tune.integration.pytorch_lightning import TuneReportCallback

from mist import utils
from mist.forward import ffn_data, ffn_model, splitter, fingerprint, ffn_train


def score_function(config, base_args, orig_dir=""):
    """score_function.

    Args:
        config: All configs passed by hyperoptimizer
        base_args: Base arguments
        orig_dir: ""
    """
    # tunedir = tune.get_trial_dir()
    # Switch s.t. we can use relative data structures
    os.chdir(orig_dir)

    kwargs = copy.deepcopy(base_args)
    kwargs.update(config)
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/paired_spectra") / dataset_name
    labels = data_dir / "labels.tsv"
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, _test_inds = splitter.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    if kwargs.get("debug", False):
        train_df = train_df[:20]
        val_df = val_df[:20]

    num_bins = kwargs.get("num_bins")
    upper_limit = kwargs.get("upper_limit")
    num_workers = kwargs.get("num_workers", 0)
    logging.info("Making datasets")

    # Remove duplicates!
    # train_df = train_df.drop_duplicates(subset="inchikey").reset_index(drop=True)
    fingerprinter = fingerprint.Fingerprinter(
        kwargs["fp_type"], dataset_name=dataset_name
    )
    train_dataset = ffn_data.BinnedDataset(
        train_df,
        data_dir=data_dir,
        num_bins=num_bins,
        upper_limit=upper_limit,
        fingerprinter=fingerprinter,
        num_workers=num_workers,
    )
    val_dataset = ffn_data.BinnedDataset(
        val_df,
        data_dir=data_dir,
        fingerprinter=fingerprinter,
        num_bins=num_bins,
        upper_limit=upper_limit,
        num_workers=num_workers,
    )
    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,  # Temp turn off shuffle
        batch_size=kwargs["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    # test_batch = next(iter(train_loader))
    logging.info("Building model")
    model = ffn_model.ForwardFFN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        output_dim=num_bins,
        use_reverse=kwargs["use_reverse"],
        learning_rate=kwargs["learning_rate"],
        decay_rate=kwargs["lr_decay_rate"],
        scheduler=kwargs["scheduler"],
        upper_limit=kwargs["upper_limit"],
        loss_name=kwargs["loss_fn"],
        fp_type=kwargs["fp_type"],
        input_dim=fingerprinter.get_nbits(),
        growing=kwargs["growing"],
        growing_weight=kwargs["growing_weight"],
        growing_layers=kwargs["growing_layers"],
        growing_scheme=kwargs["growing_scheme"],
    )

    # outputs = model(test_batch['fps'])
    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(tune.get_trial_dir(), "", ".")

    # Replace with custom callback that utilizes maximum loss during train
    tune_callback = utils.TuneReportCallback(["val_loss"])

    val_check_interval = None  # 2000 #2000
    check_val_every_n_epoch = 1

    monitor = "val_loss"
    # tb_path = tb_logger.log_dir
    earlystop_callback = EarlyStopping(monitor=monitor, patience=10)
    callbacks = [earlystop_callback, tune_callback]
    logging.info("Starting train")
    trainer = pl.Trainer(
        logger=[tb_logger],
        accelerator="gpu" if kwargs["gpu"] else None,
        devices=1 if kwargs["gpu"] else None,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)


def add_hyperopt_args(parser):
    # Tune args
    ha = parser.add_argument_group("Hyperopt Args")
    ha.add_argument("--cpus-per-trial", default=1, type=int)
    ha.add_argument("--gpus-per-trial", default=1, type=int)
    ha.add_argument("--num-h-samples", default=50, type=int)
    ha.add_argument("--grace-period", default=60 * 15, type=int)
    ha.add_argument("--max-concurrent", default=10, type=int)
    ha.add_argument("--tune-checkpoint", default=None)

    # Overwrite default savedir
    time_name = datetime.now().strftime("%Y_%m_%d")
    save_default = f"results/{time_name}_hyperopt_forward/"
    parser.set_defaults(save_dir=save_default)


def get_args():
    parser = argparse.ArgumentParser()
    ffn_train.add_forward_args(parser)
    add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial):
    """get_param_space.

    Use optuan to define this ydanmically

    """
    trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    scheduler = trial.suggest_categorical("scheduler", [True, False])
    if scheduler:
        trial.suggest_float("lr_decay_rate", 0.7, 0.999, log=True)

    trial.suggest_float("dropout", 0, 0.3, step=0.1)
    trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    trial.suggest_int("layers", 1, 3)

    growing = trial.suggest_categorical("growing", ["iterative", "none"])
    if growing != "none":
        trial.suggest_int("growing_layers", 1, 4)
        trial.suggest_categorical("growing_scheme", ["interleave"])
        trial.suggest_float("growing_weight", 1e-4, 1, log=True)


def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "learning_rate": 0.00086,
        "scheduler": False,
        "lr_decay_rate": 0.95,
        "dropout": 0.2,
        "hidden_size": 512,
        "layers": 2,
        "growing": "iterative",
        "growing_layers": 3,
        "growing_weight": 0.003,
        "growing_scheme": "interleave",
    }
    return [init_base]


def run_hyperopt():
    args = get_args()
    kwargs = args.__dict__
    ray.init()

    # Fix base_args based upon tune args
    kwargs["gpu"] = args.gpus_per_trial > 0
    max_t = args.max_epochs
    if kwargs["debug"]:
        kwargs["num_h_samples"] = 10
        kwargs["max_epochs"] = 5
    save_dir = kwargs["save_dir"]

    utils.setup_logger(
        save_dir, log_name="hyperopt.log", debug=kwargs.get("debug", False)
    )

    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Define score function
    trainable = tune.with_parameters(
        score_function, base_args=kwargs, orig_dir=Path().resolve()
    )
    yaml_args = yaml.dump(kwargs, indent=2)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    metric = "val_loss"
    # Include cpus and gpus per trial
    trainable = tune.with_resources(
        trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
    )
    search_algo = OptunaSearch(
        metric=metric,
        mode="min",
        points_to_evaluate=get_initial_points(),
        space=get_param_space,
    )
    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=args.max_concurrent)

    tuner = tune.Tuner(
        trainable,
        # param_space=param_space,
        tune_config=tune.TuneConfig(
            mode="min",
            metric=metric,
            search_alg=search_algo,
            scheduler=ASHAScheduler(
                max_t=24 * 60 * 60,  # max_t,
                time_attr="time_total_s",
                grace_period=kwargs.get("grace_period"),
                reduction_factor=2,
            ),
            num_samples=kwargs.get("num_h_samples"),
        ),
        run_config=RunConfig(name=None, local_dir=args.save_dir),
    )

    if kwargs.get("tune_checkpoint") is not None:
        ckpt = str(Path(kwargs["tune_checkpoint"]).resolve())
        tuner = tuner.restore(path=ckpt)

    results = tuner.fit()
    best_trial = results.get_best_result()
    output = {"score": best_trial.metrics[metric], "config": best_trial.config}
    out_str = yaml.dump(output, indent=2)
    logging.info(out_str)
    with open(Path(save_dir) / "best_trial.yaml", "w") as f:
        f.write(out_str)

    # Output full res table
    results.get_dataframe().to_csv(
        Path(save_dir) / "full_res_tbl.tsv", sep="\t", index=None
    )
