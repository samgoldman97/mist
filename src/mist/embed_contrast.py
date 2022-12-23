""" embed_contrast.py

This serves as the entry point to embed specs using the contrastive model

"""
import copy
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from mist.models import base
from mist.data import datasets, featurizers
from mist import utils


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True, help="Model ckpt to load from")
    parser.add_argument(
        "--dataset-name",
        required=False,
        help="Name of test dataset",
    )
    parser.add_argument("--labels-name", action="store", default="labels.tsv")
    parser.add_argument("--save-dir", required=False, default=None, help="Save dir")
    parser.add_argument("--out-name", required=False, default=None,
                        help="output name in save_dir")
    parser.add_argument(
        "--subset-datasets",
        action="store",
        default="none",
        choices=["none", "test_only"],
        help="Settings for how to subset the dataset",
    )
    parser.add_argument(
        "--num-workers", action="store", type=int, help="Get num workers", default=16
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    return parser.parse_args()


def embed_specs():
    """Run retrieval for the hdf file"""
    args = get_args()
    kwargs = args.__dict__
    dataset_name = kwargs["dataset_name"]
    debug = kwargs.get("debug")
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    max_count = 100 if debug else None
    kwargs["max_count"] = max_count
    kwargs["allow_none_smiles"] = True

    # Load saved model
    model_ckpt = kwargs.get("model_ckpt")
    pretrain_ckpt = torch.load(model_ckpt)
    main_hparams = pretrain_ckpt["hyper_parameters"]

    # Set save dir and setup model
    if kwargs.get("save_dir") is None:
        save_dir = Path(main_hparams["save_dir"]) / "embed"
        kwargs["save_dir"] = save_dir
    else:
        save_dir = kwargs["save_dir"]

    # Update main hparams with kwargs and switch to only using kwargs
    main_hparams.update(kwargs)
    kwargs = main_hparams

    # Take base model params and add these all in
    # Need to be careful beacuse these should be updated by the other args and
    # we need to avoid recursive reassignment
    base_params = kwargs["base_model_hparams"]
    kwargs["base_model_hparams"] = copy.deepcopy(base_params)
    base_params.update(kwargs)
    kwargs = base_params

    utils.setup_logger(
        save_dir, log_name=f"embed_contrast.log", debug=kwargs.get("debug", False)
    )

    # Construct model and load in state dict
    model = base.build_model(**main_hparams)
    logging.info(f"Loading from epoch {pretrain_ckpt['epoch']}")
    model.load_state_dict(pretrain_ckpt["state_dict"])
    model = model.to(device)
    model = model.eval()

    # Add spec feaatures (no mol features)
    kwargs["spec_features"] = model.main_model.spec_features(mode="test")
    kwargs["mol_features"] = "none"
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)

    # Get dataset objects
    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Subset down to appropriate names
    subset_datasets = kwargs.get("subset_datasets")
    if subset_datasets == "none":
        pass
    elif subset_datasets == "test_only":
        if kwargs.get("splitter_name") != "preset":
            raise ValueError
        split_name = Path(kwargs["split_file"])
        split_df = pd.read_csv(split_name, sep=",")
        split = sorted(list(set(split_df.keys()).difference("name")))[0]
        logging.info(f"Subset to test of split {split}")
        valid_names = set(split_df["name"][split_df[split] == "test"].values)
        spectra_mol_pairs = [
            (i, j) for i, j in spectra_mol_pairs if i.get_spec_name() in valid_names
        ]
    else:
        pass

    # Create dataset
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **kwargs
    )

    test_loader = datasets.SpecDataModule.get_paired_loader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=kwargs.get("num_workers"),
    )

    # Step 3: Encode all spectra
    encoded_specs, names = [], []
    logging.info("Encoding spectra")
    with torch.no_grad():
        model = model.to(device)
        model = model.eval()
        for spectra_batch in tqdm(test_loader):
            spectra_batch = {
                k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in spectra_batch.items()
            }

            outputs = model.encode_spectra(spectra_batch)
            contrast_out = outputs[1]["contrast"].detach().cpu()
            encoded_specs.append(contrast_out)
            names.append(spectra_batch["names"])

        names = np.concatenate(names)
        stacked_spectra = torch.cat(encoded_specs, 0).numpy()

    new_entry = {}
    new_entry["args"] = main_hparams
    new_entry["embeds"] = stacked_spectra
    new_entry["names"] = names

    # Dump to output file
    out_name = kwargs.get("output_name", None)
    if  out_name is not None:
        f_name = Path(save_dir) / out_name
    else:
        ctr = 0
        f_name = Path(save_dir) / f"embed_{dataset_name}_{ctr}.p"
        while (f_name).exists():
            ctr += 1
            f_name = save_dir / f"embed_{dataset_name}_{ctr}.p"

    # Output is new_entry
    with open(f_name, "wb") as fp:
        pickle.dump(new_entry, fp)
