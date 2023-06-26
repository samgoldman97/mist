""" train_contrastive.py

Train a contrastive model

"""
import yaml
import copy
import logging
import pickle
from pathlib import Path
import argparse
import torch

from mist import utils, parsing
from mist.models import contrastive_model
from mist.data import datasets, splitter, featurizers, data_utils


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_contrastive_args(parser)
    parsing.add_train_args(parser)
    return parser.parse_args()


def run_training():
    """run_training."""
    # Get args
    args = get_args()
    kwargs = args.__dict__
    save_dir = Path(kwargs.get("save_dir"))
    utils.setup_train(save_dir, kwargs)

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
    split_name, (train, val, test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val", "test"], [train, val, test]):
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
    test_dataset = datasets.SpectraMolMismatchHDFDataset(
        spectra_mol_list=test,
        featurizer=paired_featurizer,
        hdf_prefix=hdf_prefix,
        **kwargs,
    )

    logging.info(f"Starting fold: {split_name}")

    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, test_dataset, **kwargs
    )

    # Train the model and return list of dicts of test loss
    test_loss = model.train_model(
        spec_dataloader_module,
        log_name="",
        log_version=split_name,
        **kwargs,
    )

    # for each dict, add split name
    for j in test_loss:
        j.update({"split_name": split_name})

    # Export train dataset names
    all_train_spec_names = [
        *train_dataset.get_spectra_names(),
        *val_dataset.get_spectra_names(),
    ]
    with open(Path(model.results_dir) / "train_spec_names.p", "wb") as fp:
        pickle.dump(all_train_spec_names, fp)

    output_dict = {"args": kwargs, "results": test_loss}
    output_str = yaml.dump(output_dict, indent=2, default_flow_style=False)
    with open(save_dir / "results.yaml", "w") as fp:
        fp.write(output_str)
