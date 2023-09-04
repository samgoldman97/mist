"""hyperopt_contrastive.py

Hyperopt parameters

"""
import os
import copy
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import pytorch_lightning as pl
from ray import tune

from mist.utils import base_hyperopt
from mist import parsing
from mist.models import contrastive_model
from mist.data import datasets, splitter, featurizers


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

    new_args = copy.deepcopy(base_args)
    new_args.update(config)
    pl.utilities.seed.seed_everything(new_args.get("seed"))

    # Split data
    my_splitter = splitter.get_splitter(**new_args)

    # Get model class and build from checkpoint
    ckpt_file = new_args.get("ckpt_file")
    pretrain_ckpt = torch.load(ckpt_file)
    main_hparams = pretrain_ckpt["hyper_parameters"]
    new_args["model"] = contrastive_model.ContrastiveModel.__name__
    model = contrastive_model.ContrastiveModel(
        base_model_hparams=main_hparams, **new_args
    )


    # Load state dict from pretrained
    if not new_args.get("no_pretrain_load"):
        model.main_model.load_state_dict(pretrain_ckpt["state_dict"])

    # Use the base model args and update with any contrastive args
    kwargs = copy.deepcopy(main_hparams)
    kwargs.update(new_args)
    kwargs["dataset_type"] = model.dataset_type()

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, _test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val"], [train, val]):
        print(f"Len of {name}: {len(_data)}")

    for name, _data in zip(["train", "val"], [train, val]):
        logging.info(f"Len of {name}: {len(_data)}")

    train_dataset = datasets.SpectraMolMismatchHDFDataset(
        spectra_mol_list=train,
        featurizer=paired_featurizer,
        **kwargs,
    )
    val_dataset = datasets.SpectraMolMismatchHDFDataset(
        spectra_mol_list=val,
        featurizer=paired_featurizer,
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
        "augment_data": True,
        "weight_decay": 0.0,
        "contrastive_weight": 0.5,
    }
    return [init_base]


def run_hyperopt():
    args = get_args()
    kwargs = args.__dict__
    base_hyperopt.run_hyperopt(
        kwargs=kwargs,
        score_function=score_function,
        param_space_function=get_param_space,
        initial_points=get_initial_points(),
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_hyperopt()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
