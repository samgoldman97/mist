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
from functools import partial

from mist.models import base
from mist.data import datasets, featurizers
from mist import utils


def to_str(i):
    """ to_str. """
    if isinstance(i, bytes):
        i = i.decode()
    return str(i)


def rank_indices(
    input_dict,
    dist_fn,
    k=10,
):
    """rank_indices.

    Args:
        input_dict:
        dist_fn:
        k:
    """
    fp_embeds = input_dict["fp_embeds"]
    contrast_embed = input_dict["contrast_embed"]

    if len(fp_embeds) == 0:
        return {"inds": [], "dists": [], "ikeys": [], "smiles": []}

    sub_ikeys = input_dict["ikeys"]
    sub_smiles = input_dict["smiles"]

    dist = dist_fn(fp_embeds, contrast_embed).mean(-1)
    order = np.argsort(dist)
    offset = input_dict["offset"]

    # top k
    if k is None:
        top_k_inds = order
    else:
        top_k_inds = order[:k]

    dist = dist[top_k_inds]
    ikeys = sub_ikeys[top_k_inds]
    smiles = sub_smiles[top_k_inds]
    top_k_inds = np.array(top_k_inds) + offset
    return {"inds": top_k_inds, "dists": dist, "ikeys": ikeys, "smiles": smiles}


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
    parser.add_argument("--subform-folder", default=None, help="Subform dir")
    parser.add_argument(
        "--hdf-file",
        help="HDF file for querying retrieval",
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
    parser.add_argument( "--labels-file", help="Labels file mapping names to formulae candidates",)
    parser.add_argument("--output-tsv", default=False, action="store_true")

    parser.add_argument("--top-k", default=None, type=int)
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
    dataset_name = kwargs.get("dataset_name", "dataset")
    dist_name = kwargs.get("dist_name")
    top_k = kwargs["top_k"]
    output_tsv = kwargs.get("output_tsv", False)
    debug = kwargs.get("debug")
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    max_count = 100 if debug else None
    kwargs["max_count"] = max_count

    # Load saved model
    model_ckpt = kwargs.get("model_ckpt")
    pretrain_ckpt = torch.load(model_ckpt, map_location="cpu")
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
        split_name = Path(kwargs["split_file"])
        split_df = pd.read_csv(split_name, sep="\t")
        logging.info(f"Subset to test of split {split_name.stem}")
        valid_names = set(split_df["name"][split_df["split"] == "test"].values)
        spectra_mol_pairs = [
            (i, j) for i, j in spectra_mol_pairs if i.get_spec_name() in valid_names
        ]
    else:
        pass
    logging.info(f"Num. spec mol pairs: {len(spectra_mol_pairs)}")

    # Create dataset
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **kwargs
    )
    dist_fn = get_dist_fn(dist_name)
    rank_fn_partial = partial(rank_indices, dist_fn=dist_fn, k=top_k)
    label_to_formula = {
        i.get_spec_name(): i.get_spectra_formula()
        for i in test_dataset.get_spectra_list()
    }

    # Load hdf5 --> fp_name, retrieval_lib_name
    hdf_file = Path(kwargs["hdf_file"])
    if not Path(hdf_file).exists():
        raise ValueError(f"Cannot find hdf at path {hdf_file}")

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
    for set_inds in iterator:
        name_sub = names[set_inds]
        spec_sub = stacked_spectra[set_inds]

        # Get hdf entries
        formulas_temp = [label_to_formula.get(name) for name in name_sub]

        # Extract hdf fingerprint
        def get_decoy_fps(formula):
            with h5py.File(hdf_file, "r") as hdf_obj:
                form_ind = np.where(
                    np.array(hdf_obj["formulae"]).astype(str) == formula
                )[0]
                if len(form_ind) == 0:
                    logging.info(f"Can't find {formula} in hdf")
                    return {
                        "ikeys": [],
                        "smiles": [],
                        "fps": [],
                        "formula": formula,
                        "offset": [],
                    }

                form_ind = form_ind[0]
                length = hdf_obj["formula_lengths"][form_ind]
                offset = hdf_obj["formula_offset"][form_ind]
                sub_fps = hdf_obj["fingerprints"][offset : offset + length]
                sub_ikeys = hdf_obj["ikeys"][offset : offset + length]
                sub_smiles = hdf_obj["smiles"][offset : offset + length]
                num_bits = hdf_obj.attrs["num_bits"]

            sub_fps = utils.unpack_bits(sub_fps, num_bits)
            return {
                "ikeys": sub_ikeys,
                "smiles": sub_smiles,
                "fps": sub_fps,
                "formula": formula,
                "offset": offset,
            }

        logging.info("Extracting fingerprint for batch of size 1000")
        num_workers = kwargs.get("num_workers")
        if num_workers == 0:
            hdf_subdicts = [get_decoy_fps(form_temp) for form_temp in formulas_temp]
        else:
            hdf_subdicts = utils.chunked_parallel(
                formulas_temp, get_decoy_fps, chunks=100, max_cpu=num_workers
            )

        fp_list = [fp for i in hdf_subdicts for fp in i["fps"]]
        fp_list = np.vstack(fp_list)

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

        fp_to_embed = dict(zip([i.tobytes() for i in fp_list], stacked_mols))
        for ind, i in enumerate(hdf_subdicts):
            i["fp_embeds"] = np.array([fp_to_embed[j.tobytes()] for j in i["fps"]])
            i["contrast_embed"] = spec_sub[ind]

        if num_workers == 0 or debug:
            temp_rankings = [rank_fn_partial(i) for i in hdf_subdicts]
        else:
            temp_rankings = utils.chunked_parallel(
                hdf_subdicts,
                rank_fn_partial,
                chunks=100,
                max_cpu=kwargs.get("num_workers"),
            )
        for temp_ranking, ind_name in zip(temp_rankings, name_sub):
            temp_ranking["name"] = ind_name

        entry_rankings.extend(temp_rankings)

    new_entry = {}
    new_entry["names"] = [str(i["name"]) for i in entry_rankings]
    new_entry["ranking"] = [i["inds"] for i in entry_rankings]
    new_entry["dists"] = [i["dists"] for i in entry_rankings]
    new_entry["ikeys"] = [i["ikeys"] for i in entry_rankings]
    new_entry["smiles"] = [i["smiles"] for i in entry_rankings]
    new_entry["retrieval_settings"] = kwargs

    # Dump to output file
    f_name = (
        Path(save_dir)
        / f"retrieval_contrastive_{hdf_file.stem}_{dataset_name}_{dist_name}.p"
    )
    if (f_name).exists():
        logging.info(f"Warning: {f_name} exists and is overwritten")

    # Output is new_entry
    with open(f_name, "wb") as fp:
        pickle.dump(new_entry, fp)

    # Output tsv
    if output_tsv:
        f_name = (
            Path(save_dir) / f"retrieval_contrastive_{hdf_file.stem}_{dataset_name}_{dist_name}.tsv"
        )
        if (f_name).exists():
            logging.info(f"Warning: {f_name} exists and is overwritten")
        full_df = []
        for name, dists, rankings,  ikeys, smiles in zip(new_entry['names'], new_entry['dists'], new_entry['ranking'], new_entry['ikeys'], new_entry['smiles']): 
            form = label_to_formula.get(name)
            # Assume already sorted
            for ind, (dist, ranking, ikey, smi) in enumerate(zip(dists, rankings, 
                                                                 ikeys, smiles)):
                full_df.append({
                    "rank": ind,
                    "ikey": to_str(ikey),
                    "smi": to_str(smi), 
                    "dist": dist,
                    "name": name, 
                    "form": form,
                })
        df = pd.DataFrame(full_df).sort_values(["name", "rank"], 
                                               axis=0).reset_index(drop=True)
        df.to_csv(f_name, sep="\t", index=None)


if __name__ == "__main__":
    import time
    start_time = time.time()
    run_contrastive_retrieval()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
