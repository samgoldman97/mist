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
from mist.data import datasets, splitter, featurizers


def get_args():
    parser = argparse.ArgumentParser()
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_contrastive_args(parser)
    parsing.add_train_args(parser)
    return parser.parse_args()


def run_training():
    """run_training."""
    # Get args
    args = get_args()
    new_args = args.__dict__
    save_dir = Path(new_args.get("save_dir"))
    utils.setup_train(save_dir, new_args)

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
    split_name, (train, val, test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val", "test"], [train, val, test]):
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
    test_dataset = datasets.SpectraMolMismatchHDFDataset(
        spectra_mol_list=test,
        featurizer=paired_featurizer,
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


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_training()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
