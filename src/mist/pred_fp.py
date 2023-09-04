""" pred_fp.py

Predict fingerprints from spectra with a set of known models

"""
from pathlib import Path
import logging
import pickle
import pandas as pd
import torch
import argparse

from mist.models import base
from mist.data import datasets, featurizers
from mist import utils


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-ckpt",
        required=False,
        default=None,
        help="Model checkpoint file for prediction",
    )
    parser.add_argument(
        "--save-dir", required=False, default=None, help="Save dir option."
    )
    parser.add_argument(
        "--dataset-name",
        required=False,
        help="Name of test dataset",
    )
    parser.add_argument(
        "--num-workers", action="store", type=int, help="Get num workers", default=16
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--labels-file", action="store", default=None)
    parser.add_argument("--subform-folder", action="store", default=None)
    parser.add_argument("--spec-folder", action="store", default=None)
    parser.add_argument(
        "--output-targs",
        action="store_true",
        default=False,
        help="If true, output the targets as well",
    )
    parser.add_argument(
        "--subset-datasets",
        action="store",
        default="none",
        choices=[
            "none",
            "test_only",
        ],
        help="Settings for how to subset the dataset",
    )
    return parser.parse_args()


def run_fp_pred():
    """run_retrieval.py"""
    args = get_args()
    kwargs = args.__dict__
    debug = kwargs.get("debug")
    output_targs = kwargs.get("output_targs")
    dataset_name = kwargs.get("dataset_name", "dataset")
    model_ckpt = kwargs.get("model_ckpt")

    # Build model
    pretrain_ckpt = torch.load(model_ckpt, map_location=torch.device("cpu"))
    main_hparams = pretrain_ckpt["hyper_parameters"]

    # Replace save_dir in kwargs if it's None with the model's
    if kwargs.get("save_dir") is None:
        save_dir = Path(main_hparams["save_dir"]) / "preds"
        kwargs["save_dir"] = save_dir
    else:
        save_dir = Path(kwargs["save_dir"])

    # Update main hparams with kwargs and switch to only using kwargs
    main_hparams.update(kwargs)
    kwargs = main_hparams

    save_dir.mkdir(exist_ok=True, parents=True)
    utils.setup_logger(
        save_dir,
        log_name=f"fp_pred_{dataset_name}.log",
    )
    device = torch.device("cuda:0") if kwargs.get("gpu", False) else torch.device("cpu")

    # Hard code max_count for debugging!
    kwargs["max_count"] = 10 if debug else None

    # Create model
    model = base.build_model(**kwargs)
    logging.info(f"Loading from epoch {pretrain_ckpt['epoch']}")
    model.load_state_dict(pretrain_ckpt["state_dict"])
    model = model.to(device)
    model = model.eval()

    # Add spec feaatures (no mol features)
    kwargs["spec_features"] = model.spec_features(mode="test")
    if not output_targs:
        kwargs["mol_features"] = "none"
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Get dataset
    # Need to get a dataset where molecules can be none
    allow_none_smiles = not output_targs
    kwargs["allow_none_smiles"] = allow_none_smiles

    ## Get paired spectra
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Subset down to appropriate names
    subset_datasets = kwargs.get("subset_datasets")
    if subset_datasets == "none":
        pass
    elif subset_datasets == "test_only":
        split_name = Path(kwargs["split_file"])
        split_df = pd.read_csv(split_name, sep="\t")
        logging.info(f"Subset to test of split {split_name.stem}")
        valid_names = set(split_df["name"][split_df["split"] == "test"].values)
        spectra_mol_pairs = [
            (i, j) for i, j in spectra_mol_pairs if i.get_spec_name() in valid_names
        ]
    else:
        pass

    # Create dataset
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **kwargs
    )
    output_preds = (
        model.encode_all_spectras(test_dataset, no_grad=True, **kwargs).cpu().numpy()
    )

    output_names = test_dataset.get_spectra_names()
    targs = [None for i in output_names]
    if output_targs:
        targs = (
            model.encode_all_mols(test_dataset, no_grad=True, **kwargs)
            .cpu()
            .numpy()
            .squeeze()
        )

    split_name = Path(kwargs.get("split_file", "")).stem

    result_export = {
        "dataset_name": dataset_name,
        "names": output_names,
        "preds": output_preds,
        "targs": targs,
        "args": kwargs,
        "split_name": split_name,
    }
    results_name = f"fp_preds_{dataset_name}.p"
    save_loc = Path(save_dir).joinpath(results_name)
    with open(save_loc, "wb") as fp:
        pickle.dump(result_export, fp)


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_fp_pred()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
