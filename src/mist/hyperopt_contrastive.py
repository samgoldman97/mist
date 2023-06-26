"""hyperopt_contrastive.py

Hyperopt parameters

"""
import os
import copy
import logging
import yaml
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import pytorch_lightning as pl

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler

# from ray.tune.integration.pytorch_lightning import TuneReportCallback

from mist import utils, parsing
from mist.models import contrastive_model
from mist.data import datasets, splitter, featurizers, data_utils


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

    # Split data
    my_splitter = splitter.get_splitter(**kwargs)

    # Get model class and build from checkpoint
    ckpt_file = kwargs.get("ckpt_file")
    pretrain_ckpt = torch.load(ckpt_file)
    main_hparams = pretrain_ckpt["hyper_parameters"]
    kwargs["model"] = contrastive_model.ContrastiveModel.__name__
    model = contrastive_model.ContrastiveModel(
        base_model_hparams=main_hparams, **kwargs
    )

    # Load state dict from pretrained
    if not kwargs.get("no_pretrain_load"):
        model.main_model.load_state_dict(pretrain_ckpt["state_dict"])

    # Use the base model args and update with any contrastive args
    orig_hyperparameters = copy.copy(pretrain_ckpt["hyper_parameters"])
    orig_hyperparameters.update(kwargs)

    kwargs = orig_hyperparameters
    kwargs["dataset_type"] = model.dataset_type()

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, _test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val"], [train, val]):
        logging.info(f"Len of {name}: {len(_data)}")

    dataset_name = kwargs.get("dataset_name")
    compound_lib_name = kwargs.get("compound_lib")
    hdf_folder = (
        Path(data_utils.paired_get_spec_folder(dataset_name)).parent / "retrieval_hdf"
    )
    fp_names = "-".join(kwargs.get("fp_names"))
    hdf_prefix = str(hdf_folder / f"{compound_lib_name}_with_{fp_names}_retrieval_db")
    train_dataset = datasets.SpectraMolMismatchHDFDataset(
        spectra_mol_list=train,
        featurizer=paired_featurizer,
        hdf_prefix=hdf_prefix,
        **kwargs,
    )
    val_dataset = datasets.SpectraMolMismatchHDFDataset(
        spectra_mol_list=val,
        featurizer=paired_featurizer,
        hdf_prefix=hdf_prefix,
        **kwargs,
    )

    logging.info(f"Starting fold: {split_name}")
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, **kwargs
    )
    kwargs["save_dir"] = tune.get_trial_dir()

    # Train the model and return list of dicts of test loss
    model.train_model(
        spec_dataloader_module,
        log_name="",
        log_version=".",
        tune=True,
        **kwargs,
    )


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_contrastive_args(parser)
    parsing.add_train_args(parser)
    parsing.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial):
    """get_param_space.

    Use optuna to define this dynamically

    """

    # Training params
    trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0.0])
    scheduler = trial.suggest_categorical("scheduler", [True, False])
    if scheduler:
        trial.suggest_float("lr_decay_frac", 0.7, 0.999, log=True)
    trial.suggest_float("contrastive_weight", 0.1, 1.0, step=0.1)
    # trial.suggest_int("contrastive_scale", 1, 20)
    # trial.suggest_float("contrastive_bias", 0, 1.0, step=0.1)
    # trial.suggest_int("decoy_norm_exp", 1, 10, step=1)

    trial.suggest_float("frac_orig", 0.1, 0.8, step=0.1)
    augment_data = trial.suggest_categorical("augment_data", [True, False])
    if augment_data:
        pass


def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "learning_rate": 6e-05,
        "scheduler": False,
        "lr_decay_frac": 0.9,
        "frac_orig": 0.7,
        "augment_data": True,
        "weight_decay": 0.0,
        "contrastive_weight": 0.5,
        # "contrastive_scale": 10,
        # "contrastive_bias": 0.1,
        # "decoy_norm_exp": 4
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
        tuner = tuner.restore(path=ckpt, restart_errored=True)

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
