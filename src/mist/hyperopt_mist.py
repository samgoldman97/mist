"""hyperopt_mist.py

Hyperopt parameters

"""
import os
import copy
import logging
import yaml
import argparse
from pathlib import Path
from typing import List, Dict
from functools import partial

import pytorch_lightning as pl
import torch
from ray import tune
from ray.air import session
from tqdm import tqdm

from mist.utils import base_hyperopt
from mist import utils, parsing
from mist.models import mist_model
from mist.data import datasets, splitter, featurizers


def score_function(
    config,
    base_args,
    orig_dir="",
):
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

    # Get model class
    model_class = mist_model.MistNet
    kwargs["model"] = model_class.__name__
    kwargs["spec_features"] = model_class.spec_features()
    kwargs["mol_features"] = model_class.mol_features()
    kwargs["dataset_type"] = model_class.dataset_type()

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Build dataset
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, _test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val"], [train, val]):
        logging.info(f"Len of {name}: {len(_data)}")

    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=train, featurizer=paired_featurizer, **kwargs
    )
    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=val, featurizer=paired_featurizer, **kwargs
    )
    # Create model
    model = model_class(**kwargs)
    if kwargs.get("ckpt_file") is not None:
        model.load_from_ckpt(**kwargs)

    logging.info(f"Starting fold: {split_name}")
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, **kwargs
    )

    kwargs["save_dir"] = tune.get_trial_dir()
    torch.set_num_threads(1)

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
    parsing.add_train_args(parser)
    parsing.add_mist_args(parser)
    parsing.add_hyperopt_args(parser)
    return parser.parse_args()


def get_param_space(trial, kwargs={}):
    """get_param_space."""
    # Training params
    trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    scheduler = trial.suggest_categorical("scheduler", [True, False])
    if scheduler:
        trial.suggest_float("lr_decay_frac", 0.9, 0.999, log=True)

    form_embedder = trial.suggest_categorical(
        "form_embedder", ["rbf", "float", "pos-cos"]
    )
    trial.suggest_categorical("weight_decay", [1e-6, 1e-7, 0.0])

    # Model params
    trial.suggest_float("spectra_dropout", 0, 0.3, step=0.1)
    trial.suggest_categorical("hidden_size", [128, 256, 512])
    trial.suggest_int("peak_attn_layers", 1, 5)
    trial.suggest_categorical("augment_data", [True, False])

    # Iterative  loss
    trial.suggest_float("iterative_loss_weight", 0.0, 1, step=0.1)
    trial.suggest_int("refine_layers", 1, 5)

    # Magma loss
    augment_data = trial.suggest_categorical("augment_data", [True, False])
    trial.suggest_int("magma_loss_lambda", 0, 15)


def get_initial_points() -> List[Dict]:
    """get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    param_list = []
    init_base = {
        "learning_rate": 0.00077,
        "scheduler": False,
        "lr_decay_frac": 0.90,
        "weight_decay": 1e-7,
        "spectra_dropout": 0.1,
        "hidden_size": 256,
        "peak_attn_layers": 2,
        "augment_data": True,
        "iterative_loss_weight": 0.4,
        "refine_layers": 4,
        "form_embedder": "float",
        "magma_loss_lambda": 8,
    }

    param_list.append(init_base)
    return param_list


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
    logging.info(f"Program finished in: {end_time - start_time} seconds")
