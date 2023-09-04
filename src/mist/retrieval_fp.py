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

def to_str(i):
    """ to_str. """
    if isinstance(i, bytes):
        i = i.decode()
    return str(i)

def rank_indices(
    input_tuple,
    hdf_file_name,
    label_to_formula,
    dist_fn,
    k=10,
):
    """rank_indices.

    Args:
        input_tuple:
        hdf_file_name:
        label_to_formula:
        dist_fn:
        k:
    """
    with h5py.File(hdf_file_name, "r") as hdf_obj:
        name, pred, targ = input_tuple
        formula = label_to_formula.get(name)
        form_ind = np.where(np.array(hdf_obj["formulae"]).astype(str) == formula)[0]
        if len(form_ind) == 0:
            logging.info(f"Can't find {formula} in hdf")
            return {"inds": [], "dists": [], "ikeys": [], "smiles": []}

        form_ind = form_ind[0]
        length = hdf_obj["formula_lengths"][form_ind]
        offset = hdf_obj["formula_offset"][form_ind]
        sub_fps = hdf_obj["fingerprints"][offset : offset + length]
        sub_ikeys = hdf_obj["ikeys"][offset : offset + length]
        sub_smiles = hdf_obj["smiles"][offset : offset + length]
        num_bits = hdf_obj.attrs["num_bits"]

    # Unpack fp
    sub_fps = utils.unpack_bits(sub_fps, num_bits)
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
    parser.add_argument("--fp-pred-file", required=True, help="Pickled FP predictions")
    parser.add_argument(
        "--labels-file",
        required=False,
        help="Labels file mapping names to formulae candidates",
    )
    parser.add_argument("--save-dir", required=False, default=None, help="Save dir")
    parser.add_argument(
        "--hdf-file",
        help="HDF file to use for querying retrieval.",
        default="data/paired_spectra/csi2022/retrieval_hdf/pubchem_with_csi_retrieval_db.h5",
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


def run_retrieval():
    """Run retrieval for the hdf file"""
    args = get_args()
    kwargs = args.__dict__
    dist_name = kwargs.get("dist_name")
    debug = kwargs.get("debug")
    max_count = 100 if debug else None
    top_k = kwargs.get("top_k")
    output_tsv = kwargs.get("output_tsv", False)

    # Load fingerprints
    fp_pred_file = Path(kwargs.get("fp_pred_file"))
    entry = pickle.load(open(fp_pred_file, "rb"))
    dataset_name = entry.get("dataset_name", "dataset")

    # Set save dir and setup model
    if kwargs.get("save_dir") is None:
        save_dir = fp_pred_file.parent.parent / "retrieval"
        kwargs["save_dir"] = save_dir
    else:
        save_dir = kwargs["save_dir"]

    utils.setup_logger(save_dir, log_name=f"retrieval_fp.log", debug=debug)
    label_df = pd.read_csv(kwargs.get("labels_file"), sep="\t").astype(str)
    label_to_formula = dict(label_df[["spec", "formula"]].values)
    dist_fn = get_dist_fn(dist_name)

    # Load hdf5 --> fp_name, retrieval_lib_name
    hdf_file = Path(kwargs["hdf_file"])
    if not hdf_file.exists():
        raise ValueError(f"Cannot find hdf at path {hdf_file}")

    # If we want to speed up, don't open the hdf file in every single
    # process but rather share it
    rank_indices_parallel = partial(
        rank_indices,
        hdf_file_name=hdf_file,
        label_to_formula=label_to_formula,
        dist_fn=dist_fn,
        k=top_k,
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
    if kwargs.get("num_workers") <= 1 or debug:
        entry_ranking = [rank_indices_parallel(i) for i in input_list]
    else:
        # Get list of lists with top k
        entry_ranking = utils.chunked_parallel(
            input_list,
            rank_indices_parallel,
            chunks=100,
            max_cpu=kwargs.get("num_workers"),
        )
    new_entry = {k: v for k, v in entry.items() if k not in ["preds", "targs"]}
    new_entry["ranking"] = [i["inds"] for i in entry_ranking]
    new_entry["dists"] = [i["dists"] for i in entry_ranking]
    new_entry["ikeys"] = [i["ikeys"] for i in entry_ranking]
    new_entry["smiles"] = [i["smiles"] for i in entry_ranking]
    new_entry["retrieval_settings"] = kwargs

    # Dump to output file
    f_name = (
        Path(save_dir) / f"retrieval_fp_{hdf_file.stem}_{dataset_name}_{dist_name}.p"
    )
    if (f_name).exists():
        logging.info(f"Warning: {f_name} exists and is overwritten")

    # Output is new_entry
    with open(f_name, "wb") as fp:
        pickle.dump(new_entry, fp)

    # Output tsv
    if output_tsv:
        f_name = (
            Path(save_dir) / f"retrieval_fp_{hdf_file.stem}_{dataset_name}_{dist_name}.tsv"
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
    run_retrieval()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
