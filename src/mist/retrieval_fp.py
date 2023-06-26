""" retrieval_fp.py

Retrieval by fingerprint

"""
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import argparse
import h5py
from functools import partial

from mist import utils


def rank_indices(
    input_tuple,
    hdf_file_name,
    label_to_formula,
    pickled_indices,
    dist_fn,
    k=10,
):
    """rank_indices.

    Args:
        input_tuple:
        hdf_file_name:
        label_to_formula:
        pickled_indices:
        dist_fn:
        k:
    """
    name, pred, targ = input_tuple
    formula = label_to_formula.get(name)
    formula_dict = pickled_indices.get(formula)

    if formula_dict is None:
        logging.info(f"Can't find {formula} in hdf")
        return np.array([]), np.array([])

    offset = formula_dict["offset"]
    length = formula_dict["length"]
    hdf_file = h5py.File(hdf_file_name, "r")

    sub_fps = hdf_file["fingerprints"][offset : offset + length]

    pred_shape = pred.shape

    if len(pred_shape) == 1:
        dist = dist_fn(pred[None, :], sub_fps).sum(-1)
    elif len(pred_shape) == 2:
        # Assume we have ensemble
        dists = []
        for pred_entry in pred:
            dist = dist_fn(pred_entry[None, :], sub_fps).sum(-1)
            dists.append(dist)
        dist = np.vstack(dists).mean(0)
    else:
        raise NotImplementedError()

    order = np.argsort(dist)
    # rankings = np.argsort(order)

    # top k
    if k is not None:
        top_k_inds = order
    else:
        top_k_inds = order[:k]
    dist = dist[top_k_inds]
    top_k_inds = np.array(top_k_inds) + offset
    return top_k_inds, dist


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-pred-file", required=True, help="Pickled FP predictions")
    parser.add_argument(
        "--labels-file",
        required=False,
        help="Labels file mapping names to formulae candidates",
    )
    parser.add_argument("--save-dir", required=False, default=None, help="Save dir")
    parser.add_argument(
        "--hdf-prefix",
        help="HDF Prefix to use for querying retrieval.",
        default="data/paired_spectra/csi2022/retrieval_hdf/pubchem_with_csi_retrieval_db",
    )
    parser.add_argument(
        "--dist-name",
        help="Name of distance function",
        default="bce",
        choices=["bce", "l1", "l2", "cosine"],
    )
    parser.add_argument(
        "--num-workers", action="store", type=int, help="Get num workers", default=16
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--pred-formula",
        default=False,
        action="store_true",
        help="If true, use pred formula not true formula",
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


def run_retrieval():
    """Run retrieval for the hdf file"""
    args = get_args()
    kwargs = args.__dict__
    dist_name = kwargs.get("dist_name")
    debug = kwargs.get("debug")
    max_count = 100 if debug else None

    # Load fingerprints
    fp_pred_file = Path(kwargs.get("fp_pred_file"))
    entry = pickle.load(open(fp_pred_file, "rb"))
    dataset_name = entry["dataset_name"]

    # Set save dir and setup model
    if kwargs.get("save_dir") is None:
        save_dir = fp_pred_file.parent.parent / "retrieval"
        kwargs["save_dir"] = save_dir
    else:
        save_dir = kwargs["save_dir"]

    utils.setup_logger(save_dir, log_name=f"retrieval_fp.log", debug=debug)
    label_df = pd.read_csv(kwargs.get("labels_file"), sep="\t")
    if args.pred_formula:
        label_to_formula = dict(label_df[["spec", "pred_formula"]].values)
    else:
        label_to_formula = dict(label_df[["spec", "formula"]].values)
    dist_fn = get_dist_fn(dist_name)

    # Load hdf5 --> fp_name, retrieval_lib_name
    hdf_prefix = Path(kwargs["hdf_prefix"])
    hdf_prefix_stem = hdf_prefix.stem
    hdf_file_name = hdf_prefix.parent / f"{hdf_prefix_stem}.hdf5"
    index_file = hdf_prefix.parent / f"{hdf_prefix_stem}_index.p"
    if not hdf_file_name.exists() or not index_file.exists():
        raise ValueError(f"Cannot find hdf at path {hdf_prefix}")

    pickled_indices = pickle.load(open(index_file, "rb"))

    # If we want to speed up, don't open the hdf file in every single
    # process but rather share it
    rank_indices_parallel = partial(
        rank_indices,
        hdf_file_name=hdf_file_name,
        label_to_formula=label_to_formula,
        pickled_indices=pickled_indices,
        dist_fn=dist_fn,
        k=None,
    )

    # Conduct predictions
    preds = np.array(entry["preds"])
    targs = np.array(entry["targs"])

    # Don't require a target
    # has_targ = [i is not None for i in targs]
    # preds = preds[has_targ]
    # targs = targs[has_targ]
    names = np.array(entry["names"])  # [has_targ]

    if debug:
        preds, targs, names = preds[:max_count], targs[:max_count], names[:max_count]

    # temp_mask = names == "CCMSLIB00003137144"
    # names = names[temp_mask]
    # targs = targs[temp_mask]
    # preds = preds[temp_mask]

    # Run prediction
    input_list = list(zip(names, preds, targs))
    # entry_ranking = [parallel_read_temp(i) for i in input_list]

    # Get list of lists with top k
    entry_ranking = utils.chunked_parallel(
        input_list, rank_indices_parallel, chunks=100, max_cpu=kwargs.get("num_workers")
    )
    entry_ranking, entry_dists = zip(*entry_ranking)

    new_entry = {k: v for k, v in entry.items() if k not in ["preds", "targs"]}
    new_entry["ranking"] = entry_ranking
    new_entry["dists"] = entry_dists
    new_entry["retrieval_settings"] = kwargs

    # Dump to output file
    ctr = 0
    f_name = (
        Path(save_dir)
        / f"retrieval_fp_{hdf_prefix_stem}_{dataset_name}_{dist_name}_{ctr}.p"
    )
    while (f_name).exists():
        ctr += 1
        f_name = (
            Path(save_dir)
            / f"retrieval_fp_{hdf_prefix_stem}_{dataset_name}_{dist_name}_{ctr}.p"
        )

    # Output is new_entry
    with open(f_name, "wb") as fp:
        pickle.dump(new_entry, fp)


if __name__ == "__main__":
    run_retrieval()
