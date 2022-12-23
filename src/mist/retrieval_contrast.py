""" retrieval_contrast.py

This serves as the entry point to evaluate mass spec retrieval model

"""
import copy
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import argparse
import h5py
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
    parser.add_argument("--save-dir", required=False, default=None, help="Save dir")
    parser.add_argument("--out-name", required=False, default=None,
                        help="Save name")
    parser.add_argument(
        "--hdf-prefix",
        help="HDF Prefix to use for querying retrieval.",
        default="data/paired_spectra/csi2022/retrieval_hdf/pubchem_with_csi_retrieval_db",
    )
    parser.add_argument(
        "--dist-name",
        help="Name of distance function",
        default="cosine",
        choices=["bce", "l1", "l2", "cosine"],
    )
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
    parser.add_argument(
        "--pred-formula",
        default=False,
        action="store_true",
        help="If true, use pred formula not true formula",
    )
    parser.add_argument(
        "--labels-name",
        required=False,
        help="Labels file mapping names to formulae candidates",
    )
    return parser.parse_args()


def _bce_dist(pred, targ):
    """_bce_dist.

    Args:
        pred:
        targ:
    """
    one_term = targ * utils.clamped_log_np(pred, -5)
    zero_term = (1 - targ) * utils.clamped_log_np(1 - pred, -5)
    return -(one_term + zero_term)


def _cosine_dist(pred, targ):
    """_cosine_dist.

    Args:
        pred:
        targ:
    """

    numerator = (pred * targ).sum(-1)
    vec_norm = lambda x: (x**2).sum(-1) ** (0.5)
    denom = vec_norm(pred) * vec_norm(targ)
    cos_sim = numerator / denom

    # Expand the last dim so that it's consistent with other distances
    cos_sim = np.expand_dims(cos_sim, -1)
    return 1 - cos_sim


def get_dist_fn(dist_name):
    return {
        "bce": _bce_dist,
        "cosine": _cosine_dist,
        "l1": lambda pred, targ: np.abs(pred - targ),
        "l2": lambda pred, targ: np.square(pred - targ),
    }.get(dist_name, None)


def run_contrastive_retrieval():
    """Run retrieval for the hdf file"""
    args = get_args()
    kwargs = args.__dict__
    dataset_name = kwargs["dataset_name"]
    dist_name = kwargs.get("dist_name")
    debug = kwargs.get("debug")
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    max_count = 100 if debug else None
    kwargs["max_count"] = max_count

    # Load saved model
    model_ckpt = kwargs.get("model_ckpt")
    pretrain_ckpt = torch.load(model_ckpt)
    main_hparams = pretrain_ckpt["hyper_parameters"]

    # Set save dir and setup model
    if kwargs.get("save_dir") is None:
        save_dir = Path(main_hparams["save_dir"]) / "retrieval"
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
        save_dir, log_name=f"retrieval_hdf.log", debug=kwargs.get("debug", False)
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
    spectra_mol_pairs = datasets.get_paired_spectra(allow_none_smiles=True, **kwargs)
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
    dist_fn = get_dist_fn(dist_name)
    label_to_formula = {
        i.get_spec_name(): i.get_spectra_formula()
        for i in test_dataset.get_spectra_list()
    }

    # Load hdf5 --> fp_name, retrieval_lib_name
    hdf_prefix = Path(kwargs["hdf_prefix"])
    hdf_prefix_stem = hdf_prefix.stem
    hdf_file_name = hdf_prefix.parent / f"{hdf_prefix_stem}.hdf5"
    index_file = hdf_prefix.parent / f"{hdf_prefix_stem}_index.p"
    if not hdf_file_name.exists() or not index_file.exists():
        raise ValueError(f"Cannot find hdf at path {hdf_prefix}")

    pickled_indices = pickle.load(open(index_file, "rb"))
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

    # Break up the application of finding these
    iter_inds = list(range(len(names)))
    iterator = utils.batches(iter_inds, chunk_size=1000)
    entry_rankings = []
    entry_dists = []
    for set_inds in iterator:
        name_sub = names[set_inds]
        spec_sub = stacked_spectra[set_inds]

        # Get hdf entries
        formulas_temp = [label_to_formula.get(name) for name in name_sub]
        formula_dicts_temp = [pickled_indices.get(formula) for formula in formulas_temp]

        # Extract hdf fingerprint
        def get_decoy_fps(formula_dict):
            if formula_dict is None or len(formula_dict) == 0:
                return []
            offset, length = formula_dict.get("offset"), formula_dict.get("length")
            pubchem_hdf = h5py.File(hdf_file_name, "r")
            fps = pubchem_hdf["fingerprints"]
            outs = fps[offset : offset + length]
            pubchem_hdf.close()
            return outs

        logging.info("Extracting fingerprint  for batch of size 1000")
        num_workers = kwargs.get("num_workers")
        if num_workers == 0:
            fps_from_hdf = [get_decoy_fps(form_temp) for form_temp in formula_dicts_temp]
        else:
            fps_from_hdf = utils.chunked_parallel(
                formula_dicts_temp,
                get_decoy_fps,
                chunks=100,
                max_cpu=num_workers
            )

        # Encode all hdf fp's with single model
        fp_list = np.vstack([i for i in fps_from_hdf if len(i) > 0])

        # Get the HDF entries for each of these
        fp_loader = torch.utils.data.DataLoader(fp_list, batch_size=128, shuffle=False)
        encoded_mols = []
        logging.info("Encoding mols")
        with torch.no_grad():
            model = model.to(device)
            model = model.eval()
            for fp_batch in tqdm(fp_loader):
                batch = {"mols": fp_batch.to(device)}

                outputs = model.encode_mol(batch)
                contrast_out = outputs[1]["contrast"].detach().cpu()
                encoded_mols.append(contrast_out)
            stacked_mols = torch.cat(encoded_mols, 0).numpy()

        # Create list that is: 1. pred, offset, list
        cur_offset = 0
        tuples = []
        for name, pred, formula_dict in zip(name_sub, spec_sub, formula_dicts_temp):
            if formula_dict is None or len(formula_dict) == 0:
                tuples.append(None)
                continue

            offset, length = formula_dict["offset"], formula_dict["length"]
            mol_subset = stacked_mols[cur_offset : cur_offset + length]
            cur_offset += length
            new_tuple = (pred, mol_subset, offset)
            tuples.append(new_tuple)

        # Step 4: Conduct retrieval and do this in parallel
        def ranked_retrieval(input_tuple):
            if input_tuple is None:
                return np.array([]), np.array([])
            pred, candidates, offset = input_tuple
            dist = dist_fn(pred[None, :], candidates).sum(-1)
            order = np.argsort(dist)

            # Rankings, values
            return order + offset, dist[order]

        if num_workers == 0:
            temp_rankings = [ranked_retrieval(i) for i in tuples]
        else:
            temp_rankings = utils.chunked_parallel(
                tuples, ranked_retrieval, chunks=100, max_cpu=kwargs.get("num_workers")
            )
        temp_rankings, temp_dists = zip(*temp_rankings)
        entry_rankings.extend(temp_rankings)
        entry_dists.extend(temp_dists)

    new_entry = {}
    new_entry.update(main_hparams)
    new_entry["ranking"] = entry_rankings
    new_entry["dists"] = entry_dists
    new_entry["names"] = names
    new_entry["retrieval_settings"] = kwargs

    # Dump to output file
    ctr = 0
    save_name = kwargs.get('out_name', None)
    if save_name is not None:
        f_name = Path(save_dir) / save_name
    else:
        f_name = (
            Path(save_dir)
            / f"retrieval_contrast_{hdf_prefix_stem}_{dataset_name}_{dist_name}_{ctr}.p"
        )
        while (f_name).exists():
            ctr += 1
            f_name = (
                Path(save_dir)
                / f"retrieval_contrast_{hdf_prefix_stem}_{dataset_name}_{dist_name}_{ctr}.p"
            )

    # Output is new_entry
    with open(f_name, "wb") as fp:
        pickle.dump(new_entry, fp)
